use super::*;
use std::fs;
use std::path::Path;

#[derive(Clone, Copy)]
struct ScanSpec {
    comps: &'static [usize],
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
}

const LIBJPEG_MERGED: &[ScanSpec] = &[
    ScanSpec { comps: &[0, 1, 2], ss: 0, se: 0, ah: 0, al: 1 },
    ScanSpec { comps: &[0], ss: 1, se: 5, ah: 0, al: 2 },
    ScanSpec { comps: &[2], ss: 1, se: 63, ah: 0, al: 1 },
    ScanSpec { comps: &[1], ss: 1, se: 63, ah: 0, al: 1 },
    ScanSpec { comps: &[0], ss: 6, se: 63, ah: 0, al: 2 },
    ScanSpec { comps: &[0], ss: 1, se: 63, ah: 2, al: 1 },
    ScanSpec { comps: &[0, 1, 2], ss: 0, se: 0, ah: 1, al: 0 },
    ScanSpec { comps: &[2], ss: 1, se: 63, ah: 1, al: 0 },
    ScanSpec { comps: &[1], ss: 1, se: 63, ah: 1, al: 0 },
    ScanSpec { comps: &[0], ss: 1, se: 63, ah: 1, al: 0 },
];

const MOZJPEG_SPLIT: &[ScanSpec] = &[
    ScanSpec { comps: &[0], ss: 0, se: 0, ah: 0, al: 1 },
    ScanSpec { comps: &[1], ss: 0, se: 0, ah: 0, al: 1 },
    ScanSpec { comps: &[2], ss: 0, se: 0, ah: 0, al: 1 },
    ScanSpec { comps: &[0], ss: 1, se: 4, ah: 0, al: 1 },
    ScanSpec { comps: &[0], ss: 1, se: 4, ah: 1, al: 0 },
    ScanSpec { comps: &[1], ss: 1, se: 4, ah: 0, al: 0 },
    ScanSpec { comps: &[2], ss: 1, se: 4, ah: 0, al: 0 },
    ScanSpec { comps: &[0], ss: 0, se: 0, ah: 1, al: 0 },
    ScanSpec { comps: &[1], ss: 0, se: 0, ah: 1, al: 0 },
    ScanSpec { comps: &[2], ss: 0, se: 0, ah: 1, al: 0 },
    ScanSpec { comps: &[0], ss: 5, se: 24, ah: 0, al: 1 },
    ScanSpec { comps: &[0], ss: 5, se: 24, ah: 1, al: 0 },
    ScanSpec { comps: &[0], ss: 25, se: 63, ah: 0, al: 1 },
    ScanSpec { comps: &[0], ss: 25, se: 63, ah: 1, al: 0 },
    ScanSpec { comps: &[1], ss: 5, se: 63, ah: 0, al: 1 },
    ScanSpec { comps: &[1], ss: 5, se: 63, ah: 1, al: 0 },
    ScanSpec { comps: &[2], ss: 5, se: 63, ah: 0, al: 1 },
    ScanSpec { comps: &[2], ss: 5, se: 63, ah: 1, al: 0 },
];

const GRAY_PROG: &[ScanSpec] = &[
    ScanSpec { comps: &[0], ss: 0, se: 0, ah: 0, al: 1 },
    ScanSpec { comps: &[0], ss: 1, se: 5, ah: 0, al: 2 },
    ScanSpec { comps: &[0], ss: 6, se: 63, ah: 0, al: 2 },
    ScanSpec { comps: &[0], ss: 1, se: 63, ah: 2, al: 1 },
    ScanSpec { comps: &[0], ss: 0, se: 0, ah: 1, al: 0 },
    ScanSpec { comps: &[0], ss: 1, se: 63, ah: 1, al: 0 },
];

struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    bits: u8,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { out: Vec::new(), acc: 0, bits: 0 }
    }

    fn put_bits(&mut self, val: u16, n: u8) {
        if n == 0 {
            return;
        }
        self.acc = (self.acc << n) | u32::from(val & ((1u16 << n) - 1));
        self.bits += n;
        while self.bits >= 8 {
            self.bits -= 8;
            let b = ((self.acc >> self.bits) & 0xFF) as u8;
            self.out.push(b);
            if b == 0xFF {
                self.out.push(0x00);
            }
        }
    }

    fn pad_byte(&mut self) {
        if self.bits > 0 {
            let pad = 8 - self.bits;
            self.put_bits((1u16 << pad) - 1, pad);
        }
    }

    fn emit_rst(&mut self, rst_num: u8) {
        self.pad_byte();
        self.out.push(0xFF);
        self.out.push(0xD0 + (rst_num & 7));
    }
}

fn mag_cat(v: i16) -> (u8, u16) {
    let a = v.unsigned_abs();
    if a == 0 {
        return (0, 0);
    }
    let cat = 16 - a.leading_zeros() as u8;
    let bits = if v > 0 { a } else { a ^ ((1u16 << cat) - 1) };
    (cat, bits)
}

fn ac_huff_symbols() -> Vec<u8> {
    let mut syms = Vec::with_capacity(176);
    for r in 0..16u8 {
        for s in 0..=10u8 {
            syms.push((r << 4) | s);
        }
    }
    syms
}

fn emit_dc_huff(bw: &mut BitWriter, sym: u8) {
    // 12 symbols (0..=11) all at code length 4: codes 0b0000..=0b1011
    bw.put_bits(u16::from(sym), 4);
}

fn emit_ac_huff(bw: &mut BitWriter, sym: u8) {
    // 176 symbols: first 128 at length 8 (0..128), next 48 at length 9 (256..304)
    let idx = ((sym >> 4) as usize) * 11 + ((sym & 0x0F) as usize);
    if idx < 128 {
        bw.put_bits(idx as u16, 8);
    } else {
        bw.put_bits((256 + (idx - 128)) as u16, 9);
    }
}

fn flush_eob_run(bw: &mut BitWriter, eob_run: &mut u16, eob_ref_bits: &mut Vec<u8>) {
    if *eob_run == 0 {
        return;
    }
    let n = *eob_run;
    let r = 15 - n.leading_zeros() as u8;
    emit_ac_huff(bw, r << 4);
    if r > 0 {
        bw.put_bits(n & ((1u16 << r) - 1), r);
    }
    for &b in eob_ref_bits.iter() {
        bw.put_bits(u16::from(b), 1);
    }
    eob_ref_bits.clear();
    *eob_run = 0;
}

fn encode_progressive_jpeg(
    width: u16,
    height: u16,
    sampling: &[(u8, u8)],
    restart_interval: u16,
    scans: &[ScanSpec],
    seed: u32,
) -> Vec<u8> {
    let h_max = sampling.iter().map(|s| s.0).max().unwrap() as u16;
    let v_max = sampling.iter().map(|s| s.1).max().unwrap() as u16;
    let mcu_w = (width + h_max * 8 - 1) / (h_max * 8);
    let mcu_h = (height + v_max * 8 - 1) / (v_max * 8);

    // Build deterministic zigzag blocks [comp][by * bw + bx][zz]
    let mut blocks: Vec<Vec<[i16; 64]>> = Vec::new();
    let mut comp_dims: Vec<(u16, u16)> = Vec::new();
    let mut state = seed.wrapping_mul(0x9E37_79B9).wrapping_add(1);
    let mut next_rand = || -> u32 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };

    for (ci, &(hs, vs)) in sampling.iter().enumerate() {
        let bw = mcu_w * u16::from(hs);
        let bh = mcu_h * u16::from(vs);
        comp_dims.push((bw, bh));
        let mut c_blocks = vec![[0i16; 64]; usize::from(bw) * usize::from(bh)];
        for by in 0..usize::from(bh) {
            for bx in 0..usize::from(bw) {
                let blk = &mut c_blocks[by * usize::from(bw) + bx];
                // DC in -200..=200
                blk[0] = ((next_rand() % 401) as i16) - 200;
                let mode = (bx + by * 3 + ci + (seed as usize)) % 5;
                if mode == 0 {
                    // All-zero AC block to exercise multi-block EOBn runs!
                    continue;
                }
                for (zz, slot) in blk.iter_mut().enumerate().skip(1) {
                    if mode == 1 && (1..=19).contains(&zz) {
                        // >16 zero run to exercise ZRL (0xF0) in both first and refine passes!
                        continue;
                    }
                    let r = next_rand();
                    if r % 3 == 0 {
                        let mag = ((r >> 4) % 45) as i16 + 1;
                        *slot = if (r >> 12) & 1 == 0 { mag } else { -mag };
                    }
                }
            }
        }
        blocks.push(c_blocks);
    }

    let mut jpg = Vec::new();
    // SOI
    jpg.extend_from_slice(&[0xFF, 0xD8]);
    // APP0 JFIF (ensures standard YCbCr color transform for 3-component images)
    if sampling.len() == 3 {
        jpg.extend_from_slice(&[
            0xFF, 0xE0, 0x00, 0x10, b'J', b'F', b'I', b'F', 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x00,
        ]);
    }
    // DQT table 0 & 1
    for qt_id in 0..2u8 {
        jpg.extend_from_slice(&[0xFF, 0xDB, 0x00, 67, qt_id]);
        for i in 0..64u8 {
            jpg.push(2 + ((i.wrapping_mul(3).wrapping_add(qt_id * 5)) % 14));
        }
    }
    // DRI
    if restart_interval > 0 {
        jpg.extend_from_slice(&[0xFF, 0xDD, 0x00, 0x04]);
        jpg.extend_from_slice(&restart_interval.to_be_bytes());
    }
    // SOF2 (Progressive DCT)
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
        jpg.push(if ci == 0 { 0 } else { 1 });
    }
    // DHT: DC table 0
    jpg.extend_from_slice(&[0xFF, 0xC4, 0x00, 19 + 12, 0x00]);
    let mut dc_counts = [0u8; 16];
    dc_counts[3] = 12;
    jpg.extend_from_slice(&dc_counts);
    for s in 0..12u8 {
        jpg.push(s);
    }
    // DHT: AC table 0
    let ac_syms = ac_huff_symbols();
    jpg.extend_from_slice(&[0xFF, 0xC4]);
    jpg.extend_from_slice(&(19 + ac_syms.len() as u16).to_be_bytes());
    jpg.push(0x10);
    let mut ac_counts = [0u8; 16];
    ac_counts[7] = 128;
    ac_counts[8] = 48;
    jpg.extend_from_slice(&ac_counts);
    jpg.extend_from_slice(&ac_syms);

    // Scans
    for scan in scans {
        let ns = scan.comps.len();
        let sos_len = (6 + 2 * ns) as u16;
        jpg.extend_from_slice(&[0xFF, 0xDA]);
        jpg.extend_from_slice(&sos_len.to_be_bytes());
        jpg.push(ns as u8);
        for &ci in scan.comps {
            jpg.push((ci + 1) as u8);
            jpg.push(0x00);
        }
        jpg.push(scan.ss);
        jpg.push(scan.se);
        jpg.push((scan.ah << 4) | scan.al);

        let is_interleaved = ns > 1;
        let (max_mx, max_my) = if is_interleaved {
            (mcu_w, mcu_h)
        } else {
            comp_dims[scan.comps[0]]
        };

        let mut bw = BitWriter::new();
        let mut dc_pred = [0i16; 4];
        let mut eob_run = 0u16;
        let mut eob_ref_bits: Vec<u8> = Vec::new();
        let mut mcus_left = restart_interval;
        let mut rst_num = 0u8;

        for my in 0..max_my {
            if my * 8 >= height {
                break;
            }
            for mx in 0..max_mx {
                if mx * 8 >= width {
                    break;
                }
                if restart_interval > 0 {
                    if mcus_left == 0 {
                        flush_eob_run(&mut bw, &mut eob_run, &mut eob_ref_bits);
                        bw.emit_rst(rst_num);
                        rst_num = (rst_num + 1) & 7;
                        dc_pred = [0; 4];
                        mcus_left = restart_interval;
                    }
                    mcus_left -= 1;
                }

                for (slot_i, &ci) in scan.comps.iter().enumerate() {
                    let (hs, vs) = if is_interleaved {
                        (u16::from(sampling[ci].0), u16::from(sampling[ci].1))
                    } else {
                        (1, 1)
                    };
                    let comp_bw = usize::from(comp_dims[ci].0);
                    for vp in 0..vs {
                        for hp in 0..hs {
                            let bx = usize::from(mx * hs + hp);
                            let by = usize::from(my * vs + vp);
                            let blk = &blocks[ci][by * comp_bw + bx];

                            if scan.ss == 0 {
                                if scan.ah == 0 {
                                    let shifted = blk[0] >> scan.al;
                                    let diff = shifted - dc_pred[slot_i];
                                    dc_pred[slot_i] = shifted;
                                    let (cat, bits) = mag_cat(diff);
                                    emit_dc_huff(&mut bw, cat);
                                    bw.put_bits(bits, cat);
                                } else {
                                    let bit = ((blk[0] >> scan.al) & 1) as u16;
                                    bw.put_bits(bit, 1);
                                }
                            } else if scan.ah == 0 {
                                let mut zrun = 0u8;
                                for zz in scan.ss..=scan.se {
                                    let c = blk[usize::from(zz)];
                                    let mag = c.unsigned_abs() >> scan.al;
                                    if mag == 0 {
                                        zrun += 1;
                                    } else {
                                        flush_eob_run(&mut bw, &mut eob_run, &mut eob_ref_bits);
                                        while zrun >= 16 {
                                            emit_ac_huff(&mut bw, 0xF0);
                                            zrun -= 16;
                                        }
                                        let v = if c > 0 { mag as i16 } else { -(mag as i16) };
                                        let (cat, bits) = mag_cat(v);
                                        emit_ac_huff(&mut bw, (zrun << 4) | cat);
                                        bw.put_bits(bits, cat);
                                        zrun = 0;
                                    }
                                }
                                if zrun > 0 {
                                    eob_run += 1;
                                    if eob_run == 0x7FFF {
                                        flush_eob_run(&mut bw, &mut eob_run, &mut eob_ref_bits);
                                    }
                                }
                            } else {
                                // AC refinement scan (ah == al + 1)
                                #[derive(Clone, Copy)]
                                enum AcRef {
                                    Zero,
                                    PrevNz(u8),
                                    NewNz(u8),
                                }
                                let mut items = Vec::with_capacity(64);
                                let mut last_new = None;
                                for (idx, zz) in (scan.ss..=scan.se).enumerate() {
                                    let c = blk[usize::from(zz)];
                                    let abs = c.unsigned_abs();
                                    if (abs >> scan.ah) != 0 {
                                        items.push(AcRef::PrevNz(((abs >> scan.al) & 1) as u8));
                                    } else if ((abs >> scan.al) & 1) != 0 {
                                        items.push(AcRef::NewNz(u8::from(c > 0)));
                                        last_new = Some(idx);
                                    } else {
                                        items.push(AcRef::Zero);
                                    }
                                }
                                let emit_end = last_new.map_or(0, |p| p + 1);
                                if emit_end > 0 {
                                    flush_eob_run(&mut bw, &mut eob_run, &mut eob_ref_bits);
                                }
                                let mut zrun = 0u8;
                                let mut ref_buf: Vec<u8> = Vec::new();
                                for &item in &items[..emit_end] {
                                    match item {
                                        AcRef::PrevNz(b) => ref_buf.push(b),
                                        AcRef::Zero => {
                                            zrun += 1;
                                            if zrun == 16 {
                                                emit_ac_huff(&mut bw, 0xF0);
                                                for &rb in &ref_buf {
                                                    bw.put_bits(u16::from(rb), 1);
                                                }
                                                ref_buf.clear();
                                                zrun = 0;
                                            }
                                        }
                                        AcRef::NewNz(sign) => {
                                            emit_ac_huff(&mut bw, (zrun << 4) | 1);
                                            bw.put_bits(u16::from(sign), 1);
                                            for &rb in &ref_buf {
                                                bw.put_bits(u16::from(rb), 1);
                                            }
                                            ref_buf.clear();
                                            zrun = 0;
                                        }
                                    }
                                }
                                if emit_end < items.len() {
                                    eob_run += 1;
                                    for &item in &items[emit_end..] {
                                        if let AcRef::PrevNz(b) = item {
                                            eob_ref_bits.push(b);
                                        }
                                    }
                                    if eob_run == 0x7FFF {
                                        flush_eob_run(&mut bw, &mut eob_run, &mut eob_ref_bits);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        flush_eob_run(&mut bw, &mut eob_run, &mut eob_ref_bits);
        bw.pad_byte();
        jpg.extend_from_slice(&bw.out);
    }

    // EOI
    jpg.extend_from_slice(&[0xFF, 0xD9]);
    jpg
}

fn assert_scaled_matches_unpatched(bytes: &[u8], context: &str) {
    let mut probe = Decoder::new(bytes);
    probe.read_info().unwrap();
    let info = probe.info().unwrap();
    assert_eq!(info.coding_process, CodingProcess::DctProgressive);

    let (w, h) = (info.width, info.height);
    let requests = [
        (w, h),
        ((w + 1) / 2, (h + 1) / 2),
        ((w + 3) / 4, (h + 3) / 4),
        ((w + 7) / 8, (h + 7) / 8),
        (cmp::max(1, w / 3), cmp::max(1, h / 3)),
        (1, 1),
    ];

    for (rw, rh) in requests {
        let mut patched = Decoder::new(bytes);
        let p_dims = patched.scale(rw, rh).unwrap();
        let p_pixels = patched.decode().unwrap();
        let p_info = patched.info().unwrap();

        let mut unpatched = Decoder::new(bytes);
        unpatched.force_unscaled_coeffs = true;
        let u_dims = unpatched.scale(rw, rh).unwrap();
        let u_pixels = unpatched.decode().unwrap();
        let u_info = unpatched.info().unwrap();

        assert_eq!(p_dims, u_dims, "{}: scale({},{}) dims mismatch", context, rw, rh);
        assert_eq!(p_info, u_info, "{}: scale({},{}) info mismatch", context, rw, rh);
        assert_eq!(p_pixels, u_pixels, "{}: scale({},{}) pixel mismatch", context, rw, rh);
    }
}

fn collect_progressive_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_progressive_files(&path, out);
        } else if matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("jpg") | Some("jpeg")
        ) {
            let bytes = fs::read(&path).unwrap();
            let mut dec = Decoder::new(bytes.as_slice());
            if dec.read_info().is_ok()
                && dec.info().map(|i| i.coding_process) == Some(CodingProcess::DctProgressive)
            {
                out.push(path);
            }
        }
    }
}

const CMYK_PROG: &[ScanSpec] = &[
    ScanSpec { comps: &[0, 1, 2, 3], ss: 0, se: 0, ah: 0, al: 1 },
    ScanSpec { comps: &[0], ss: 1, se: 5, ah: 0, al: 1 },
    ScanSpec { comps: &[0], ss: 6, se: 63, ah: 0, al: 1 },
    ScanSpec { comps: &[0], ss: 1, se: 63, ah: 1, al: 0 },
    ScanSpec { comps: &[1], ss: 1, se: 63, ah: 0, al: 0 },
    ScanSpec { comps: &[2], ss: 1, se: 63, ah: 0, al: 0 },
    ScanSpec { comps: &[3], ss: 1, se: 63, ah: 0, al: 1 },
    ScanSpec { comps: &[3], ss: 1, se: 63, ah: 1, al: 0 },
    ScanSpec { comps: &[0, 1, 2, 3], ss: 0, se: 0, ah: 1, al: 0 },
];

#[test]
fn reftest_progressive_corpus_matches_unpatched_at_all_scales() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("reftest")
        .join("images");
    let mut files = Vec::new();
    collect_progressive_files(&root, &mut files);
    files.sort();
    assert!(files.len() >= 6);

    for path in files {
        let bytes = fs::read(&path).unwrap();
        assert_scaled_matches_unpatched(&bytes, &path.display().to_string());
    }
}

#[test]
fn progressive_multi_chunk_sizes_match_unpatched() {
    // 525x533 at 1/2 scale spans two coefficient chunks with a partial tail chunk.
    let subsamplings: &[(&str, &[(u8, u8)])] = &[
        ("4:4:4", &[(1, 1), (1, 1), (1, 1)]),
        ("4:2:0", &[(2, 2), (1, 1), (1, 1)]),
    ];
    let mut seed = 7_001u32;
    for &(sub_name, sampling) in subsamplings {
        for &dri in &[0u16, 7] {
            for &(script_name, script) in &[("libjpeg-merged", LIBJPEG_MERGED), ("mozjpeg-split", MOZJPEG_SPLIT)] {
                let bytes = encode_progressive_jpeg(525, 533, sampling, dri, script, seed);
                seed = seed.wrapping_add(1);
                let ctx = format!("{} 525x533 dri={} {}", sub_name, dri, script_name);
                assert_scaled_matches_unpatched(&bytes, &ctx);
            }
        }
    }
}

#[test]
fn progressive_scales_subsamplings_odd_dims_and_restarts_match_unpatched() {
    let subsamplings: &[(&str, &[(u8, u8)])] = &[
        ("4:4:4", &[(1, 1), (1, 1), (1, 1)]),
        ("4:2:2", &[(2, 1), (1, 1), (1, 1)]),
        ("4:2:0", &[(2, 2), (1, 1), (1, 1)]),
        ("4:4:0", &[(1, 2), (1, 1), (1, 1)]),
        ("4:2:1-intermediate", &[(4, 1), (2, 1), (1, 1)]),
        ("grayscale", &[(1, 1)]),
        ("cmyk-221122", &[(2, 2), (1, 1), (1, 1), (2, 2)]),
    ];
    let dims: &[(u16, u16)] = &[
        (1, 1),
        (7, 5),
        (9, 11),
        (16, 16),
        (19, 27),
        (33, 45),
        (67, 53),
    ];
    let restarts: &[u16] = &[0, 1, 7];

    let mut seed = 1u32;
    for &(sub_name, sampling) in subsamplings {
        let scripts: &[(&str, &[ScanSpec])] = match sampling.len() {
            1 => &[("gray", GRAY_PROG)],
            4 => &[("cmyk", CMYK_PROG)],
            _ => &[("libjpeg-merged", LIBJPEG_MERGED), ("mozjpeg-split", MOZJPEG_SPLIT)],
        };
        for &(w, h) in dims {
            for &dri in restarts {
                for &(script_name, script) in scripts {
                    let bytes = encode_progressive_jpeg(w, h, sampling, dri, script, seed);
                    seed = seed.wrapping_add(1);
                    let ctx = format!("{} {}x{} dri={} {}", sub_name, w, h, dri, script_name);
                    assert_scaled_matches_unpatched(&bytes, &ctx);
                }
            }
        }
    }
}
