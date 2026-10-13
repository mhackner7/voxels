#[repr(C, u8)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
pub enum Command {
    Player(PlayerCommands),
    Camera(CameraCommands),
    Inventory(InventoryCommands),
    Developer(DeveloperCommands),
    NULL,
}

impl Command {
    /// get the command from specified index number
    pub const LUT: [Self; Self::MAX_COUNT] = Self::build_lookup_table();

    /// number of total commands
    pub const MAX_COUNT: usize = PlayerCommands::_COUNT as usize
        + CameraCommands::_COUNT as usize
        + InventoryCommands::_COUNT as usize
        + DeveloperCommands::_COUNT as usize;

    /*
     useful offsets when converting
    */
    const ENTITY_OFFSET: usize = 0;
    pub const ENTITY_RANGE: std::ops::Range<usize> = Self::ENTITY_OFFSET..Self::CAMERA_OFFSET;

    const CAMERA_OFFSET: usize = PlayerCommands::_COUNT as usize;
    pub const CAMERA_RANGE: std::ops::Range<usize> = Self::CAMERA_OFFSET..Self::INVENTORY_OFFSET;

    const INVENTORY_OFFSET: usize = Self::CAMERA_OFFSET + CameraCommands::_COUNT as usize;
    pub const INVENTORY_RANGE: std::ops::Range<usize> =
        Self::INVENTORY_OFFSET..Self::DEVELOPER_OFFSET;

    const DEVELOPER_OFFSET: usize = Self::INVENTORY_OFFSET + InventoryCommands::_COUNT as usize;
    pub const DEVELOPER_RANGE: std::ops::Range<usize> = Self::DEVELOPER_OFFSET..Self::MAX_COUNT;

    /// gets the index value of the enum member
    /// i.e. EntityCommands::MoveBackward is 1
    #[inline(always)]
    pub const fn index(self) -> usize {
        match self {
            Self::Player(c) => c as usize + Self::ENTITY_OFFSET,
            Self::Camera(c) => c as usize + Self::CAMERA_OFFSET,
            Self::Inventory(c) => c as usize + Self::INVENTORY_OFFSET,
            Self::Developer(c) => c as usize + Self::DEVELOPER_OFFSET,
            Command::NULL => panic!("You are trying to index with COMMANDS::NULL, fix the code"),
        }
    }

    const fn build_lookup_table() -> [Self; Self::MAX_COUNT] {
        let mut arr = [Self::NULL; Self::MAX_COUNT];

        let mut i = 0;
        while i < Self::MAX_COUNT {
            arr[i] = if i >= Self::DEVELOPER_OFFSET {
                Command::Developer(unsafe {
                    std::mem::transmute((i - Self::DEVELOPER_OFFSET) as u8)
                })
            } else if i >= Self::INVENTORY_OFFSET {
                Command::Inventory(unsafe {
                    std::mem::transmute((i - Self::INVENTORY_OFFSET) as u8)
                })
            } else if i >= Self::CAMERA_OFFSET {
                Command::Camera(unsafe { std::mem::transmute((i - Self::CAMERA_OFFSET) as u8) })
            } else {
                Command::Player(unsafe { std::mem::transmute(i as u8) })
            };

            i += 1;
        }

        arr
    }
}

pub trait SubCommand: Copy {
    const COUNT: usize;
    fn as_usize(self) -> usize;
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
/// commands that pertain to entity/player controls
pub enum PlayerCommands {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    Sprint,
    Jump,
    Sneak,
    Destroy,
    Interact,
    Select,
    Inventory,
    BrushGrow,
    BrushShrink,
    _COUNT,
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
/// commands that pertain to camera controls
pub enum CameraCommands {
    ZoomIn,
    ZoomOut,
    ChangePerspective,
    _COUNT,
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
///commands that pertain to Inventory related commands
pub enum InventoryCommands {
    Select,
    Next,
    Back,
    Split,
    QuickMove,
    _COUNT,
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, PartialOrd, Ord, Eq)]
/// commands that pertain to developer and debug tools, as well as display changes
/// will contain global important keybinds as well: Pause, Back
pub enum DeveloperCommands {
    Pause,
    Back,
    VolumeUp,
    VolumeDown,
    OpenDevMenu,
    ShowDebugInfoLight,
    ShowDebugInfoHeavy,
    ForceMinimize,
    ToggleFullscreen,
    _COUNT,
}

impl From<Command> for &'static str {
    fn from(value: Command) -> Self {
        match value {
            Command::Player(cmd) => match cmd {
                PlayerCommands::MoveForward => "MoveForward",
                PlayerCommands::MoveBackward => "MoveBackward",
                PlayerCommands::MoveLeft => "MoveLeft",
                PlayerCommands::MoveRight => "MoveRight",
                PlayerCommands::Sprint => "Sprint",
                PlayerCommands::Jump => "Jump",
                PlayerCommands::Sneak => "Sneak",
                PlayerCommands::Destroy => "Destroy",
                PlayerCommands::Interact => "Interact",
                PlayerCommands::Select => "Select",
                PlayerCommands::Inventory => "Inventory",
                PlayerCommands::BrushGrow => "BrushGrow",
                PlayerCommands::BrushShrink => "BrushShrink",
                PlayerCommands::_COUNT => "Entity::_COUNT",
            },
            Command::Camera(cmd) => match cmd {
                CameraCommands::ZoomIn => "ZoomIn",
                CameraCommands::ZoomOut => "ZoomOut",
                CameraCommands::ChangePerspective => "ChangePerspective",
                CameraCommands::_COUNT => "Camera::_COUNT",
            },
            Command::Inventory(cmd) => match cmd {
                InventoryCommands::Select => "Select",
                InventoryCommands::Next => "Next",
                InventoryCommands::Back => "Back",
                InventoryCommands::Split => "Split",
                InventoryCommands::QuickMove => "QuickMove",
                InventoryCommands::_COUNT => "Inventory::_COUNT",
            },
            Command::Developer(cmd) => match cmd {
                DeveloperCommands::Pause => "Pause",
                DeveloperCommands::Back => "Back",
                DeveloperCommands::VolumeUp => "VolumeUp",
                DeveloperCommands::VolumeDown => "VolumeDown",
                DeveloperCommands::OpenDevMenu => "OpenDevMenu",
                DeveloperCommands::ShowDebugInfoLight => "ShowDebugInfoLight",
                DeveloperCommands::ShowDebugInfoHeavy => "ShowDebugInfoHeavy",
                DeveloperCommands::ForceMinimize => "ForceMinimize",
                DeveloperCommands::ToggleFullscreen => "ToggleFullscreen",
                DeveloperCommands::_COUNT => "Developer::_COUNT",
            },
            Command::NULL => "NULL",
        }
    }
}

impl From<PlayerCommands> for Command {
    fn from(value: PlayerCommands) -> Self {
        Command::Player(value)
    }
}
impl From<CameraCommands> for Command {
    fn from(value: CameraCommands) -> Self {
        Command::Camera(value)
    }
}
impl From<InventoryCommands> for Command {
    fn from(value: InventoryCommands) -> Self {
        Command::Inventory(value)
    }
}
impl From<DeveloperCommands> for Command {
    fn from(value: DeveloperCommands) -> Self {
        Command::Developer(value)
    }
}

impl From<Command> for usize {
    fn from(value: Command) -> Self {
        value.index()
    }
}

impl std::fmt::Display for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str: &'static str = (*self).into();
        write!(f, "{}", str)
    }
}

impl SubCommand for PlayerCommands {
    const COUNT: usize = PlayerCommands::_COUNT as usize;
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl SubCommand for CameraCommands {
    const COUNT: usize = CameraCommands::_COUNT as usize;
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl SubCommand for InventoryCommands {
    const COUNT: usize = InventoryCommands::_COUNT as usize;
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl SubCommand for DeveloperCommands {
    const COUNT: usize = DeveloperCommands::_COUNT as usize;

    fn as_usize(self) -> usize {
        self as usize
    }
}
