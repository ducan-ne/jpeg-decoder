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

fn minimal_baseline_zero_jpeg(width: u16, height: u16, sampling: &[(u8, u8)]) -> Vec<u8> {
    let h_max = sampling.iter().map(|s| s.0).max().unwrap() as usize;
    let v_max = sampling.iter().map(|s| s.1).max().unwrap() as usize;
    let mcu_w = (usize::from(width) + h_max * 8 - 1) / (h_max * 8);
    let mcu_h = (usize::from(height) + v_max * 8 - 1) / (v_max * 8);
    let blocks_per_mcu: usize = sampling
        .iter()
        .map(|&(hs, vs)| usize::from(hs) * usize::from(vs))
        .sum();
    let total_blocks = mcu_w * mcu_h * blocks_per_mcu;

    let mut jpg = Vec::new();
    jpg.extend_from_slice(&[0xFF, 0xD8]);
    // DQT 0
    jpg.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, 0x00]);
    jpg.extend_from_slice(&[8u8; 64]);
    // SOF0
    let sof_len = 8 + 3 * sampling.len() as u16;
    jpg.extend_from_slice(&[0xFF, 0xC0]);
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
    // DHT DC 0: symbol 0 -> 1-bit code '0'
    let mut counts = [0u8; 16];
    counts[0] = 1;
    jpg.extend_from_slice(&[0xFF, 0xC4, 0x00, 20, 0x00]);
    jpg.extend_from_slice(&counts);
    jpg.push(0x00);
    // DHT AC 0: symbol 0x00 (EOB) -> 1-bit code '0'
    jpg.extend_from_slice(&[0xFF, 0xC4, 0x00, 20, 0x10]);
    jpg.extend_from_slice(&counts);
    jpg.push(0x00);
    // Interleaved SOS (each block is 2 zero bits: DC=0, AC=EOB)
    let ns = sampling.len();
    let sos_len = (6 + 2 * ns) as u16;
    jpg.extend_from_slice(&[0xFF, 0xDA]);
    jpg.extend_from_slice(&sos_len.to_be_bytes());
    jpg.push(ns as u8);
    for ci in 0..ns {
        jpg.push((ci + 1) as u8);
        jpg.push(0x00);
    }
    jpg.extend_from_slice(&[0x00, 63, 0x00]);
    jpg.resize(jpg.len() + (total_blocks * 2 + 7) / 8, 0x00);
    jpg.extend_from_slice(&[0xFF, 0xD9]);
    jpg
}

fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x1000_0000_01B3);
    }
    h
}

#[test]
fn decode_rows_avoids_full_output_buffer() {
    // 1. Baseline 4096x4096 4:2:0 scaled to 1/2 (2048x2048 RGB24):
    //    component planes = 4 MiB + 1 MiB + 1 MiB = 6 MiB, full RGB24 output = 12 MiB.
    let base_jpg = minimal_baseline_zero_jpeg(4096, 4096, &[(2, 2), (1, 1), (1, 1)]);
    let full_rgb_bytes = 2048usize * 2048 * 3; // 12 MiB

    let ((full_hash, full_len), full_peak) = measure_peak_delta(|| {
        let mut dec = jpeg::Decoder::new(base_jpg.as_slice());
        assert_eq!(dec.scale(2048, 2048).unwrap(), (2048, 2048));
        let pixels = dec.decode().unwrap();
        (fnv1a(0xcbf2_9ce4_8422_2325, &pixels), pixels.len())
    });
    assert_eq!(full_len, full_rgb_bytes);
    assert!(full_peak >= 18 * 1024 * 1024);

    let ((row_hash, row_bytes, rows, base_live_in_cb), row_peak) = measure_peak_delta(|| {
        let base_live = CURRENT_BYTES.load(Ordering::SeqCst);
        let mut dec = jpeg::Decoder::new(base_jpg.as_slice());
        assert_eq!(dec.scale(2048, 2048).unwrap(), (2048, 2048));
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut bytes = 0usize;
        let mut rows = 0usize;
        let mut max_live = 0usize;
        dec.decode_rows(|row| {
            let live = CURRENT_BYTES.load(Ordering::SeqCst).saturating_sub(base_live);
            if live > max_live {
                max_live = live;
            }
            hash = fnv1a(hash, row);
            bytes += row.len();
            rows += 1;
        })
        .unwrap();
        (hash, bytes, rows, max_live)
    });
    assert_eq!(rows, 2048);
    assert_eq!(row_bytes, full_rgb_bytes);
    assert_eq!(row_hash, full_hash);
    // Only the 6 MiB component planes + single-row buffers are live during row delivery
    // (vs 6 MiB planes + 12 MiB RGB24 output = 18 MiB in decode()).
    assert!(
        base_live_in_cb < 6656 * 1024,
        "baseline 1/2 live heap in on_row {} bytes exceeded 6.5 MiB (no 12 MiB RGB24 buffer)",
        base_live_in_cb
    );
    assert!(
        full_peak >= row_peak + 6 * 1024 * 1024,
        "expected decode() peak ({}) to exceed decode_rows() peak ({}) by >= 6 MiB",
        full_peak,
        row_peak
    );

    // 2. Progressive 4096x4096 4:2:0 scaled to 1/4 (1024x1024 RGB24):
    //    1/4-scale coeffs = 3.0 MiB, planes = 1.5 MiB, full RGB24 = 3.0 MiB.
    let mut prog_jpg = minimal_progressive_header(4096, 4096, &[(2, 2), (1, 1), (1, 1)]);
    for &(comp_id, blks) in &[(1u8, 512 * 512), (2, 256 * 256), (3, 256 * 256)] {
        append_zero_dc_scan(&mut prog_jpg, comp_id, blks);
        append_eob14_ac_scan(&mut prog_jpg, comp_id, 1, 63, blks);
    }
    prog_jpg.extend_from_slice(&[0xFF, 0xD9]);
    let prog_rgb_bytes = 1024usize * 1024 * 3; // 3 MiB

    let (prog_full_hash, prog_full_peak) = measure_peak_delta(|| {
        let mut dec = jpeg::Decoder::new(prog_jpg.as_slice());
        assert_eq!(dec.scale(1024, 1024).unwrap(), (1024, 1024));
        let pixels = dec.decode().unwrap();
        assert_eq!(pixels.len(), prog_rgb_bytes);
        fnv1a(0xcbf2_9ce4_8422_2325, &pixels)
    });
    assert!(prog_full_peak >= 4500 * 1024);

    let ((prog_row_hash, prog_rows, max_live_in_callback), prog_row_peak) = measure_peak_delta(|| {
        let base_live = CURRENT_BYTES.load(Ordering::SeqCst);
        let mut dec = jpeg::Decoder::new(prog_jpg.as_slice());
        assert_eq!(dec.scale(1024, 1024).unwrap(), (1024, 1024));
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut rows = 0usize;
        let mut max_live = 0usize;
        dec.decode_rows(|row| {
            let live = CURRENT_BYTES.load(Ordering::SeqCst).saturating_sub(base_live);
            if live > max_live {
                max_live = live;
            }
            hash = fnv1a(hash, row);
            rows += 1;
        })
        .unwrap();
        (hash, rows, max_live)
    });
    assert_eq!(prog_rows, 1024);
    assert_eq!(prog_row_hash, prog_full_hash);
    // During row delivery, coeff chunks (3 MiB) are already freed and no 3 MiB RGB24 output
    // buffer exists: only the 1.5 MiB scaled YCbCr planes + single-row scratch are live.
    assert!(
        max_live_in_callback < 2 * 1024 * 1024,
        "progressive 1/4 live heap in on_row {} bytes exceeded 2 MiB (expected ~1.5 MiB planes only)",
        max_live_in_callback
    );
    assert!(
        prog_row_peak < 6 * 1024 * 1024,
        "progressive 1/4 decode_rows peak ({}) exceeded 6 MiB",
        prog_row_peak
    );
}
