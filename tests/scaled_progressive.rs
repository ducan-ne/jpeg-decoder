extern crate jpeg_decoder as jpeg;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct PeakAlloc;

static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static SERIALIZE: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let cur = CURRENT_BYTES.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK_BYTES.fetch_max(cur, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            let cur = CURRENT_BYTES.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK_BYTES.fetch_max(cur, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                let diff = new_size - layout.size();
                let cur = CURRENT_BYTES.fetch_add(diff, Ordering::SeqCst) + diff;
                PEAK_BYTES.fetch_max(cur, Ordering::SeqCst);
            } else {
                CURRENT_BYTES.fetch_sub(layout.size() - new_size, Ordering::SeqCst);
            }
        }
        new_ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT_BYTES.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc;

fn measure_peak_delta<F: FnOnce() -> R, R>(f: F) -> (R, usize) {
    let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let base = CURRENT_BYTES.load(Ordering::SeqCst);
    PEAK_BYTES.store(base, Ordering::SeqCst);
    let res = f();
    let peak = PEAK_BYTES.load(Ordering::SeqCst);
    (res, peak.saturating_sub(base))
}

fn minimal_progressive_header(width: u16, height: u16, sampling: &[(u8, u8)]) -> Vec<u8> {
    let mut jpg = Vec::new();
    jpg.extend_from_slice(&[0xFF, 0xD8]);
    // DQT 0
    jpg.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0x00]);
    jpg.extend_from_slice(&[8u8; 64]);
    // SOF2
    let sof_len = 8 + 3 * sampling.len() as u16;
    jpg.extend_from_slice(&[0xFF, 0xC2]);
    jpg.extend_from_slice(&sof_len.to_be_bytes());
    jpg.push(8);
    jpg.extend_from_slice(&height.to_be_bytes());
    jpg.extend_from_slice(&width.to_be_bytes());
    jpg.push(sampling.len() as u8);
    for (ci, &(hs, vs)) in sampling.iter().enumerate() {
        jpg.push((ci + 1) as u8);
        jpg.push((hs << 4) | vs);
        jpg.push(0);
    }
    // DHT DC 0 (symbol 0 -> 1-bit code '0')
    jpg.extend_from_slice(&[0xFF, 0xC4, 0x00, 20, 0x00]);
    let mut counts = [0u8; 16];
    counts[0] = 1;
    jpg.extend_from_slice(&counts);
    jpg.push(0x00);
    // DHT AC 0 (symbol 0xE0 = EOB14 -> 1-bit code '0')
    jpg.extend_from_slice(&[0xFF, 0xC4, 0x00, 20, 0x10]);
    jpg.extend_from_slice(&counts);
    jpg.push(0xE0);
    jpg
}

fn append_eob14_ac_scan(jpg: &mut Vec<u8>, comp_id: u8, ss: u8, se: u8, block_count: usize) {
    jpg.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, comp_id, 0x00, ss, se, 0x00]);
    // Each EOB14 symbol ('0' + 14 ones = 15 bits) skips 32,767 blocks!
    let runs = (block_count + 32766) / 32767;
    let total_bits = runs * 15;
    let mut acc = 0u32;
    let mut bits = 0u8;
    for _ in 0..runs {
        // '0' followed by 14 ones = 0x3FFF in 15 bits
        acc = (acc << 15) | 0x3FFF;
        bits += 15;
        while bits >= 8 {
            bits -= 8;
            let b = ((acc >> bits) & 0xFF) as u8;
            jpg.push(b);
            if b == 0xFF {
                jpg.push(0x00);
            }
        }
    }
    if bits > 0 {
        let pad = 8 - bits;
        let b = (((acc << pad) | ((1u32 << pad) - 1)) & 0xFF) as u8;
        jpg.push(b);
        if b == 0xFF {
            jpg.push(0x00);
        }
    }
    let _ = total_bits;
}

fn append_zero_dc_scan(jpg: &mut Vec<u8>, comp_id: u8, block_count: usize) {
    jpg.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, comp_id, 0x00, 0, 0, 0x00]);
    // Symbol 0 is 1-bit '0', so `block_count` zero bits!
    jpg.resize(jpg.len() + (block_count + 7) / 8, 0x00);
}

#[test]
fn progressive_coeff_bytes_exact_for_all_scales_and_subsamplings() {
    // 5600x5600 4:4:4 -> 700x700 blocks per component * 3 components = 1,470,000 blocks
    let hdr_444 = minimal_progressive_header(5600, 5600, &[(1, 1), (1, 1), (1, 1)]);
    let blocks_444 = 700u64 * 700 * 3;

    let mut dec = jpeg::Decoder::new(hdr_444.as_slice());
    assert_eq!(dec.progressive_coeff_bytes(), 0);
    dec.read_info().unwrap();
    assert_eq!(dec.progressive_coeff_bytes(), blocks_444 * 128);

    dec.scale(2800, 2800).unwrap();
    assert_eq!(dec.progressive_coeff_bytes(), blocks_444 * 40);

    dec.scale(1400, 1400).unwrap();
    assert_eq!(dec.progressive_coeff_bytes(), blocks_444 * 16);

    dec.scale(700, 700).unwrap();
    assert_eq!(dec.progressive_coeff_bytes(), blocks_444 * 2);

    // 6240x4160 4:2:0 -> MCU 390x260; Y: 780x520=405,600 blocks, Cb/Cr: 101,400 each -> 608,400 blocks
    let hdr_420 = minimal_progressive_header(6240, 4160, &[(2, 2), (1, 1), (1, 1)]);
    let blocks_420 = 608_400u64;
    let mut dec420 = jpeg::Decoder::new(hdr_420.as_slice());
    dec420.read_info().unwrap();
    assert_eq!(dec420.progressive_coeff_bytes(), blocks_420 * 128);
    dec420.scale(780, 520).unwrap();
    assert_eq!(dec420.progressive_coeff_bytes(), blocks_420 * 2);

    // Extreme 65535x65535 4-component header must not overflow u64
    let hdr_max = minimal_progressive_header(65535, 65535, &[(1, 1), (1, 1), (1, 1), (1, 1)]);
    let mut dec_max = jpeg::Decoder::new(hdr_max.as_slice());
    dec_max.read_info().unwrap();
    assert_eq!(dec_max.progressive_coeff_bytes(), 8192u64 * 8192 * 4 * 128);
}

#[test]
fn scan_bomb_returns_error_without_large_allocation() {
    // 4096x4096 4:4:4 (full coeff buffer would be 96 MiB) with 105 skipped/minimal scans
    let mut bomb = minimal_progressive_header(4096, 4096, &[(1, 1), (1, 1), (1, 1)]);
    let blocks_per_comp = 512 * 512;
    for _ in 0..105 {
        append_eob14_ac_scan(&mut bomb, 1, 6, 63, blocks_per_comp);
    }
    bomb.extend_from_slice(&[0xFF, 0xD9]);

    let (res, peak_delta) = measure_peak_delta(|| {
        let mut dec = jpeg::Decoder::new(bomb.as_slice());
        dec.scale(512, 512).unwrap();
        dec.decode()
    });

    match res {
        Err(jpeg::Error::Format(msg)) => assert!(msg.contains("too many scans"), "got: {}", msg),
        other => panic!("expected Format(too many scans), got {:?}", other),
    }
    assert!(
        peak_delta < 512 * 1024,
        "scan bomb allocated {} bytes (expected < 512 KiB)",
        peak_delta
    );
}

#[test]
fn truncated_progressive_returns_error_without_large_allocation() {
    // 6240x4160 4:4:4 (full coeffs = 156 MiB, eager 1/2-scale coeffs = 49 MiB)
    // Truncated after 24 bytes of the first scan!
    let mut truncated = minimal_progressive_header(6240, 4160, &[(1, 1), (1, 1), (1, 1)]);
    truncated.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0, 0, 0x00]);
    truncated.extend_from_slice(&[0u8; 24]);

    for &(rw, rh) in &[(780u16, 520u16), (1560, 1040), (3120, 2080)] {
        let (res, peak_delta) = measure_peak_delta(|| {
            let mut dec = jpeg::Decoder::new(truncated.as_slice());
            dec.scale(rw, rh).unwrap();
            dec.decode()
        });
        assert!(res.is_err(), "expected error on truncated progressive stream");
        assert!(
            peak_delta < 1024 * 1024,
            "truncated decode at scale({},{}) allocated {} bytes (expected < 1 MiB)",
            rw,
            rh,
            peak_delta
        );
    }
}

#[test]
fn scaled_progressive_decode_uses_bounded_memory() {
    // Valid 2048x2048 4:4:4 progressive JPEG: unpatched allocates 24 MiB of [i16; 64] coeffs.
    // At 1/8 scale (256x256 output), scaled coeffs are 3 * 256 * 256 * 2 B = 384 KiB + 196 KiB RGB!
    let blocks = 256 * 256;
    let mut jpg = minimal_progressive_header(2048, 2048, &[(1, 1), (1, 1), (1, 1)]);
    for comp_id in 1..=3u8 {
        append_zero_dc_scan(&mut jpg, comp_id, blocks);
        append_eob14_ac_scan(&mut jpg, comp_id, 1, 63, blocks);
    }
    jpg.extend_from_slice(&[0xFF, 0xD9]);

    let (pixels, peak_delta) = measure_peak_delta(|| {
        let mut dec = jpeg::Decoder::new(jpg.as_slice());
        let dims = dec.scale(256, 256).unwrap();
        assert_eq!(dims, (256, 256));
        dec.decode().unwrap()
    });
    assert_eq!(pixels.len(), 256 * 256 * 3);
    assert!(
        peak_delta < 2 * 1024 * 1024,
        "1/8-scale 2048x2048 4:4:4 progressive decode allocated {} bytes (expected < 2 MiB vs 24 MiB unpatched)",
        peak_delta
    );
}

#[test]
fn half_scale_progressive_decode_uses_bounded_memory() {
    // Valid 2048x2048 4:4:4 progressive JPEG at 1/2 scale (1024x1024 output): scaled coeffs and
    // nonzero masks are 3 * 65536 blocks * 40 B = 7.5 MiB, RGB output 3 MiB, vs 24 MiB unpatched.
    let blocks = 256 * 256;
    let mut jpg = minimal_progressive_header(2048, 2048, &[(1, 1), (1, 1), (1, 1)]);
    for comp_id in 1..=3u8 {
        append_zero_dc_scan(&mut jpg, comp_id, blocks);
        append_eob14_ac_scan(&mut jpg, comp_id, 1, 63, blocks);
    }
    jpg.extend_from_slice(&[0xFF, 0xD9]);

    let (pixels, peak_delta) = measure_peak_delta(|| {
        let mut dec = jpeg::Decoder::new(jpg.as_slice());
        let dims = dec.scale(1024, 1024).unwrap();
        assert_eq!(dims, (1024, 1024));
        dec.decode().unwrap()
    });
    assert_eq!(pixels.len(), 1024 * 1024 * 3);
    assert!(
        peak_delta < 14 * 1024 * 1024,
        "1/2-scale 2048x2048 4:4:4 progressive decode allocated {} bytes (expected < 14 MiB vs 24 MiB unpatched coefficients)",
        peak_delta
    );
}
