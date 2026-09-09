use std::ops::Add;

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Debug)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    _padd: u8,
}

#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord, Debug, Default)]
pub struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

#[inline]
pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Rgba {
    Rgba { r, g, b, a }
}

#[inline]
pub const fn rgb(r: u8, g: u8, b: u8) -> Rgb {
    Rgb::new(r, g, b)
}

impl Rgb {
    #[inline]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, _padd: 0 }
    }

    #[inline]
    pub fn with_red(mut self, red: u8) -> Self {
        self.r = red;
        self
    }

    #[inline]
    pub fn with_green(mut self, green: u8) -> Self {
        self.g = green;
        self
    }

    #[inline]
    pub fn with_blue(mut self, blue: u8) -> Self {
        self.b = blue;
        self
    }

    #[inline]
    pub fn compact(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, 0])
    }

    #[inline]
    pub fn compact_with_alpha(self, alpha: u8) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, alpha])
    }
}

impl Add for Rgb {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
}

impl Rgba {
    #[inline]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    #[inline]
    pub fn with_red(mut self, red: u8) -> Self {
        self.r = red;
        self
    }

    #[inline]
    pub fn with_green(mut self, green: u8) -> Self {
        self.g = green;
        self
    }

    #[inline]
    pub fn with_blue(mut self, blue: u8) -> Self {
        self.b = blue;
        self
    }

    #[inline]
    pub fn with_alpha(mut self, alpha: u8) -> Self {
        self.a = alpha;
        self
    }

    #[inline]
    pub fn compact(self) -> u32 {
        u32::from_le_bytes([self.r, self.g, self.b, self.a])
    }
}

impl Add for Rgba {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.r + rhs.r,
            self.g + rhs.g,
            self.b + rhs.b,
            self.a + rhs.a,
        )
    }
}
