use super::*;
use crate::engine::input::commands::*;
use crate::simd::packetray::Packet;
use crate::simd::packetvec::Packed_Vec3_x4;
use crate::simd::simdf32::SimdF32;
use crate::simd::simdu32::SimdU32;
use crate::world::leaf::{Brick, Leaf};
use crate::world::tree64::Branch;
use crate::world::voxel::*;
use std::fmt::Write;

/*
    DEBUG UI

    gets drawn straight into the pixel buffer after the render, and scales up with the render rows
    1 = debug overlay, 2 = performance overlay, 3 = dev menu (W/S pick, A/D change, 3 / esc close)
    all the text gets formatted on the stack, so nothing allocates per frame
*/

/// how many frames the frame time graph remembers
pub const FRAME_HISTORY: usize = 120;

pub struct Ui {
    /// 1: where you are, where you're looking, what you're looking at
    pub overlay: bool,
    /// 2: frame times, cpu, ram and tree stats
    pub perf: bool,
    /// 3: dev menu, it steals the movement keys while it's open
    pub menu: bool,
    /// which menu row is highlighted
    pub selected: usize,
    /// ui got drawn into the buffer last frame, so it has to be cleaned up
    pub drawn: bool,
    /// the last FRAME_HISTORY frame times (ms), as a ring
    pub frame_ms: [f32; FRAME_HISTORY],
    /// where the next frame time goes
    pub frame_cursor: usize,
}

impl Ui {
    pub const fn new() -> Self {
        Self {
            overlay: false,
            perf: false,
            menu: false,
            selected: 0,
            drawn: false,
            frame_ms: [0.0; FRAME_HISTORY],
            frame_cursor: 0,
        }
    }

    #[inline(always)]
    pub fn visible(&self) -> bool {
        self.overlay || self.perf || self.menu
    }

    /// one more frame time for the graph (ms)
    #[inline(always)]
    pub fn record(&mut self, ms: f32) {
        self.frame_ms[self.frame_cursor] = ms;
        self.frame_cursor = (self.frame_cursor + 1) % FRAME_HISTORY;
    }
}

/// dev menu rows, the values live on the engine / player / world
const MENU: [&str; 7] = [
    "COLLISION",
    "BOUNCE LIGHT",
    "RENDER ROWS",
    "FPS TARGET",
    "MOVE SPEED",
    "SUN TURN",
    "SUN HEIGHT",
];

/// names of Engine::FRAME_BUDGETS
const FPS_TARGETS: [&str; 3] = ["60", "30", "OFF"];

/// move speeds the dev menu cycles through
const MOVE_SPEEDS: [f32; 4] = [1.0, 2.0, 4.0, 8.0];

/// pixel buffer colors, 0xAABBGGRR
const WHITE: u32 = 0xFFFF_FFFF;
const GREY: u32 = 0xFFB4_B4B4;
const YELLOW: u32 = 0xFF40_E0FF;
const GREEN: u32 = 0xFF40_D040;
const RED: u32 = 0xFF40_40FF;

/// glyph cell in unscaled pixels, 5x7 glyph + 1 column / 2 rows of spacing
const CELL_X: usize = 6;
const CELL_Y: usize = 9;

/// render rows per ui scale step, so 720 rows draws the ui at 2x
const ROWS_PER_SCALE: usize = 360;

/// graph height in unscaled pixels, the top of it is 2 frames at 60 fps
const GRAPH_HEIGHT: usize = 40;
const GRAPH_MS: f32 = 2000.0 / 60.0;

/// 5x7 glyphs for ascii 32 ' ' to 96 '`', one row per byte, bit 4 is the left column
/// lowercase draws as uppercase, anything else as '?'
#[rustfmt::skip]
const FONT: [[u8; 7]; 65] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // ' '
    [0x04, 0x04, 0x04, 0x04, 0x00, 0x00, 0x04], // !
    [0x0A, 0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00], // "
    [0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A], // #
    [0x04, 0x0F, 0x14, 0x0E, 0x05, 0x1E, 0x04], // $
    [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03], // %
    [0x0C, 0x12, 0x14, 0x08, 0x15, 0x12, 0x0D], // &
    [0x0C, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00], // '
    [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02], // (
    [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08], // )
    [0x00, 0x04, 0x15, 0x0E, 0x15, 0x04, 0x00], // *
    [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00], // +
    [0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08], // ,
    [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00], // -
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C], // .
    [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00], // /
    [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E], // 0
    [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E], // 1
    [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F], // 2
    [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E], // 3
    [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02], // 4
    [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E], // 5
    [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E], // 6
    [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08], // 7
    [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E], // 8
    [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C], // 9
    [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00], // :
    [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x04, 0x08], // ;
    [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02], // <
    [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00], // =
    [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08], // >
    [0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04], // ?
    [0x0E, 0x11, 0x01, 0x0D, 0x15, 0x15, 0x0E], // @
    [0x0E, 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11], // A
    [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E], // B
    [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E], // C
    [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C], // D
    [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F], // E
    [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10], // F
    [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F], // G
    [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11], // H
    [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E], // I
    [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C], // J
    [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11], // K
    [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F], // L
    [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11], // M
    [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11], // N
    [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E], // O
    [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10], // P
    [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D], // Q
    [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11], // R
    [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E], // S
    [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04], // T
    [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E], // U
    [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04], // V
    [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A], // W
    [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11], // X
    [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04], // Y
    [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F], // Z
    [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E], // [
    [0x00, 0x10, 0x08, 0x04, 0x02, 0x01, 0x00], // \
    [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E], // ]
    [0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00], // ^
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F], // _
    [0x08, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00], // `
];

/// one line of text, formatted on the stack
#[derive(Clone, Copy)]
struct Line {
    text: [u8; 48],
    len: usize,
    color: u32,
}

impl Line {
    const EMPTY: Self = Self {
        text: [0; 48],
        len: 0,
        color: WHITE,
    };
}

impl Write for Line {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        // too long just gets cut off, never an error //
        let n = s.len().min(self.text.len() - self.len);
        self.text[self.len..self.len + n].copy_from_slice(&s.as_bytes()[..n]);
        self.len += n;
        Ok(())
    }
}

/// a block of lines drawn on one darkened panel
struct Lines {
    lines: [Line; 12],
    count: usize,
}

impl Lines {
    const fn new() -> Self {
        Self {
            lines: [Line::EMPTY; 12],
            count: 0,
        }
    }

    /// start a new empty line and hand it back for writing,
    /// a full block just keeps reusing its last line
    fn line(&mut self, color: u32) -> &mut Line {
        self.count = (self.count + 1).min(self.lines.len());

        let line = &mut self.lines[self.count - 1];
        line.len = 0;
        line.color = color;
        line
    }

    fn push(&mut self, color: u32, args: std::fmt::Arguments) {
        let _ = self.line(color).write_fmt(args);
    }

    /// panel size in unscaled pixels
    fn size(&self) -> (usize, usize) {
        let longest = self.lines[..self.count]
            .iter()
            .map(|l| l.len)
            .max()
            .unwrap_or(0);
        (longest * CELL_X + 6, self.count * CELL_Y + 4)
    }
}

impl Engine {
    /// ui toggles and dev menu navigation, runs once the input got processed
    pub fn ui_input(&mut self) {
        let activity = &self.input.activity;
        let pressed = |cmd: Command| activity.was_pressed(cmd);

        let overlay = pressed(DeveloperCommands::ShowDebugInfoLight.into());
        let perf = pressed(DeveloperCommands::ShowDebugInfoHeavy.into());
        let menu = pressed(DeveloperCommands::OpenDevMenu.into());
        let back = pressed(DeveloperCommands::Back.into());
        let up = pressed(PlayerCommands::MoveForward.into());
        let down = pressed(PlayerCommands::MoveBackward.into());
        let left = pressed(PlayerCommands::MoveLeft.into());
        let right = pressed(PlayerCommands::MoveRight.into());

        self.ui.overlay ^= overlay;
        self.ui.perf ^= perf;

        if menu || (back && self.ui.menu) {
            self.ui.menu = !self.ui.menu;
            return;
        }

        if !self.ui.menu {
            return;
        }

        // menu navigation //

        if up {
            self.ui.selected = (self.ui.selected + MENU.len() - 1) % MENU.len();
        }

        if down {
            self.ui.selected = (self.ui.selected + 1) % MENU.len();
        }

        for (pressed, step) in [(left, -1), (right, 1)] {
            if pressed {
                self.menu_change(step);
            }
        }
    }

    /// nudge the value of the selected menu row
    fn menu_change(&mut self, step: i32) {
        match self.ui.selected {
            0 => self.player.noclip = !self.player.noclip,
            1 => {
                self.bounce_lighting = !self.bounce_lighting;
                self.restart_image();
            }
            2 => {
                self.render_rows = cycle(&Self::RENDER_ROWS, self.render_rows, step);
                self.resize_render = true;
            }
            3 => self.fps_target = (self.fps_target as i32 + step).rem_euclid(3) as usize,
            4 => self.player.move_speed = cycle(&MOVE_SPEEDS, self.player.move_speed, step),
            row => {
                // the sun lives in the world now, moving it restarts the image //

                let sun = self.world.sun;
                let (turn, raise) = if row == 5 { (15.0, 0.0) } else { (0.0, 5.0) };

                self.world.set_sun(
                    sun.azimuth + turn * step as f32,
                    sun.elevation + raise * step as f32,
                );
                self.restart_image();
            }
        }
    }

    /// overlays, crosshair and menu, all on top of the rendered image
    pub fn draw_ui(&mut self, buffer_address: usize) {
        self.ui.drawn = self.ui.visible();

        if !self.ui.drawn {
            return;
        }

        let (width, height) = (self.render_width, self.render_height);
        let scale = (height / ROWS_PER_SCALE).max(1);

        // SAFETY: the pixel buffer holds render_width * render_height pixels
        // and the render is done writing it
        let pixels =
            unsafe { std::slice::from_raw_parts_mut(buffer_address as *mut u32, width * height) };

        /* DEBUG OVERLAY */

        if self.ui.overlay {
            let mut lines = Lines::new();
            let c = self.camera.center;

            lines.push(
                WHITE,
                format_args!("POS {:.1} {:.1} {:.1}", c[0], c[1], c[2]),
            );
            lines.push(
                WHITE,
                format_args!(
                    "YAW {:.0}  PITCH {:.0}",
                    self.camera.yaw.to_degrees().rem_euclid(360.0),
                    self.camera.pitch.to_degrees()
                ),
            );

            // whatever voxel is under the crosshair, one ray straight down the view //

            let packet = Packet::new(
                Packed_Vec3_x4::from_single(c),
                Packed_Vec3_x4::from_single(self.camera.front),
            );
            let hit = self.world.tree.trace::<false, false>(
                packet,
                SimdU32::ALL_ACTIVE,
                SimdF32::INFY,
                0.0,
            );

            if hit.hit.has_active() {
                let t = hit.distances.to_array()[0];
                let n = [hit.normals.xs, hit.normals.ys, hit.normals.zs].map(|v| v.to_array()[0]);

                // half a voxel back against the normal is inside the hit voxel //
                let voxel = |a: usize| {
                    ((c[a] + self.camera.front[a] * t - n[a] * 0.5 * VOXEL_SIZE) * VOXELS_PER_METER)
                        .floor() as i32
                };

                lines.push(
                    WHITE,
                    format_args!(
                        "LOOK {:?} {:.1}M  AT {} {} {}",
                        hit.materials[0],
                        t,
                        voxel(0),
                        voxel(1),
                        voxel(2)
                    ),
                );
            } else {
                lines.push(WHITE, format_args!("LOOK SKY"));
            }

            let player = &self.player;
            let state = &player.state;

            lines.push(
                WHITE,
                format_args!(
                    "{} {:?} {:?} {:?}  HAND {:?}",
                    player.name, state.y_state, state.gait, state.medium, player.hand
                ),
            );
            lines.push(
                WHITE,
                format_args!(
                    "{}  SPEED {}X  BOOST {:.1}X",
                    if player.noclip && player.flying() {
                        "NOCLIP"
                    } else {
                        "COLLISION"
                    },
                    player.move_speed,
                    player.fly_boost
                ),
            );
            lines.push(
                WHITE,
                format_args!(
                    "SUN {:.0} {:.0}  BOUNCE {}",
                    self.world.sun.azimuth,
                    self.world.sun.elevation,
                    on_off(self.bounce_lighting)
                ),
            );
            lines.push(GREY, format_args!("1 DEBUG  2 PERF  3 MENU"));

            draw_lines(pixels, width, 2, 2, &lines, scale);
            draw_crosshair(pixels, width, height, scale);
        }

        /* PERFORMANCE OVERLAY */

        if self.ui.perf {
            self.usage.update();

            let mut lines = Lines::new();
            let usage = &self.usage;
            let tree = &self.world.tree;
            const MB: f64 = 1024.0 * 1024.0;

            let last = self.ui.frame_ms[(self.ui.frame_cursor + FRAME_HISTORY - 1) % FRAME_HISTORY];

            lines.push(
                WHITE,
                format_args!("FPS {:.0}  FRAME {:.1}MS", self.smooth_fps, last),
            );
            lines.push(
                WHITE,
                format_args!(
                    "RENDER {:.1}MS  PRESENT {:.1}MS",
                    self.render_ms, self.present_ms
                ),
            );
            lines.push(
                WHITE,
                format_args!(
                    "RES {}X{}  TARGET {}  MOVING 1/{} {}X  {} SLICES",
                    width,
                    height,
                    FPS_TARGETS[self.fps_target],
                    Self::MOVING_STEPS[self.moving_step].0,
                    Self::MOVING_STEPS[self.moving_step].1,
                    self.still_slices
                ),
            );
            lines.push(
                WHITE,
                format_args!(
                    "SAMPLES {:.0}/{}",
                    self.samples,
                    Self::MAX_ACCUMULATED_FRAMES
                ),
            );

            if usage.total == 0 {
                lines.push(WHITE, format_args!("CPU N/A  RAM N/A"));
            } else {
                lines.push(
                    WHITE,
                    format_args!(
                        "CPU {:.0}%  {:.1} OF {} CORES",
                        100.0 * usage.cores_busy / usage.cores as f32,
                        usage.cores_busy,
                        usage.cores
                    ),
                );
                lines.push(
                    WHITE,
                    format_args!(
                        "RAM {:.0}MB OF {:.0}MB",
                        usage.resident as f64 / MB,
                        usage.total as f64 / MB
                    ),
                );
            }

            let tree_bytes = tree.branches.len() * size_of::<Branch>()
                + tree.leaves.len() * (size_of::<Leaf>() + size_of::<Brick>() + 16);

            lines.push(
                GREY,
                format_args!(
                    "TREE {:.1}MB  {} BRANCHES  {} LEAVES",
                    tree_bytes as f64 / MB,
                    tree.branches.len(),
                    tree.leaves.len()
                ),
            );

            // text above the graph, both tucked in the bottom left //

            let graph_y = height.saturating_sub((GRAPH_HEIGHT + 2) * scale);
            let text_y = graph_y.saturating_sub(lines.size().1 * scale + scale);

            draw_lines(pixels, width, 2, text_y, &lines, scale);
            draw_graph(pixels, width, 2, graph_y, &self.ui, scale);
        }

        /* MENU */

        if self.ui.menu {
            let mut lines = Lines::new();

            lines.push(WHITE, format_args!("DEV MENU"));

            for (row, name) in MENU.iter().enumerate() {
                let (cursor, color) = if row == self.ui.selected {
                    ('>', YELLOW)
                } else {
                    (' ', WHITE)
                };

                let line = lines.line(color);
                let _ = write!(line, "{cursor} {name:<13}");

                let _ = match row {
                    0 => line.write_str(on_off(!self.player.noclip)),
                    1 => line.write_str(on_off(self.bounce_lighting)),
                    2 => write!(line, "{}", self.render_rows),
                    3 => line.write_str(FPS_TARGETS[self.fps_target]),
                    4 => write!(line, "{}X", self.player.move_speed),
                    5 => write!(line, "{:.0}", self.world.sun.azimuth),
                    _ => write!(line, "{:.0}", self.world.sun.elevation),
                };
            }

            lines.push(GREY, format_args!("W/S SELECT  A/D CHANGE  3 CLOSE"));

            let (panel_width, _) = lines.size();
            draw_lines(
                pixels,
                width,
                width.saturating_sub(panel_width * scale + 2),
                2,
                &lines,
                scale,
            );
        }
    }
}

#[inline(always)]
fn on_off(value: bool) -> &'static str {
    if value { "ON" } else { "OFF" }
}

/// the option `step` spots after current, wrapping around
fn cycle<T: PartialEq + Copy>(options: &[T], current: T, step: i32) -> T {
    let index = options.iter().position(|&o| o == current).unwrap_or(0) as i32;
    options[(index + step).rem_euclid(options.len() as i32) as usize]
}

/// halve every channel of a rectangle (pixels), keep alpha, clipped to the buffer
fn darken(pixels: &mut [u32], width: usize, x: usize, y: usize, w: usize, h: usize) {
    let height = pixels.len() / width;

    for py in y..(y + h).min(height) {
        for p in &mut pixels[py * width + x.min(width)..py * width + (x + w).min(width)] {
            *p = ((*p >> 1) & 0x007F_7F7F) | 0xFF00_0000;
        }
    }
}

/// fill a rectangle (pixels), clipped to the buffer
fn fill(pixels: &mut [u32], width: usize, x: usize, y: usize, w: usize, h: usize, color: u32) {
    let height = pixels.len() / width;

    for py in y..(y + h).min(height) {
        pixels[py * width + x.min(width)..py * width + (x + w).min(width)].fill(color);
    }
}

/// darkened panel at (x, y) pixels with every line on it, glyphs scaled up by scale
fn draw_lines(pixels: &mut [u32], width: usize, x: usize, y: usize, lines: &Lines, scale: usize) {
    let (panel_width, panel_height) = lines.size();
    darken(
        pixels,
        width,
        x,
        y,
        panel_width * scale,
        panel_height * scale,
    );

    for (row, line) in lines.lines[..lines.count].iter().enumerate() {
        let line_y = y + (3 + row * CELL_Y) * scale;

        for (column, &byte) in line.text[..line.len].iter().enumerate() {
            let glyph = match byte.to_ascii_uppercase() {
                b @ 32..=96 => &FONT[(b - 32) as usize],
                _ => &FONT[(b'?' - 32) as usize],
            };

            let glyph_x = x + (3 + column * CELL_X) * scale;

            for (gy, bits) in glyph.iter().enumerate() {
                for gx in 0..5 {
                    if bits >> (4 - gx) & 1 != 0 {
                        let (px, py) = (glyph_x + gx * scale, line_y + gy * scale);
                        fill(pixels, width, px, py, scale, scale, line.color);
                    }
                }
            }
        }
    }
}

/// frame time bars, oldest on the left, with a line at 60 fps
fn draw_graph(pixels: &mut [u32], width: usize, x: usize, y: usize, ui: &Ui, scale: usize) {
    let h = GRAPH_HEIGHT * scale;
    darken(pixels, width, x, y, FRAME_HISTORY * scale, h);

    for i in 0..FRAME_HISTORY {
        let ms = ui.frame_ms[(ui.frame_cursor + i) % FRAME_HISTORY];
        let bar = ((ms / GRAPH_MS).min(1.0) * h as f32) as usize;

        let color = if ms <= 1000.0 / 60.0 {
            GREEN
        } else if ms <= GRAPH_MS {
            YELLOW
        } else {
            RED
        };

        fill(pixels, width, x + i * scale, y + h - bar, scale, bar, color);
    }

    // 60 fps sits half way up //
    fill(pixels, width, x, y + h / 2, FRAME_HISTORY * scale, 1, GREY);
}

/// small plus at the screen center
fn draw_crosshair(pixels: &mut [u32], width: usize, height: usize, scale: usize) {
    let (cx, cy) = (width / 2, height / 2);

    for d in 2..6 {
        let d = d * scale;
        fill(pixels, width, cx - d, cy, scale, scale, WHITE);
        fill(pixels, width, cx + d, cy, scale, scale, WHITE);
        fill(pixels, width, cx, cy - d, scale, scale, WHITE);
        fill(pixels, width, cx, cy + d, scale, scale, WHITE);
    }
}
