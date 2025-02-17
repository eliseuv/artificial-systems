use std::{
    fmt::Display,
    time::{Duration, Instant},
};

/// Print debug info only in debug mode
/// https://users.rust-lang.org/t/show-value-only-in-debug-mode/43686/5
#[macro_export]
macro_rules! dbg {
    ($($x:tt)*) => {
        {
            #[cfg(debug_assertions)]
            {
                std::dbg!($($x)*)
            }
            #[cfg(not(debug_assertions))]
            {
                ($($x)*)
            }
        }
    }
}

/// Simple timer
#[derive(Debug)]
pub struct Timer {
    description: String,
    start: Instant,
}

impl Timer {
    /// Create new timer
    #[inline(always)]
    pub fn new(description: &str) -> Self {
        Self {
            description: description.to_owned(),
            start: Instant::now(),
        }
    }

    /// Start/Restart the timer
    #[inline(always)]
    pub fn start(&mut self) {
        self.start = Instant::now();
    }

    /// Read elapsed time
    #[inline(always)]
    pub fn read(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Display for Timer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[Timer] {description}: {time_secs}s",
            description = self.description,
            time_secs = self.read().as_secs_f64()
        )
    }
}
