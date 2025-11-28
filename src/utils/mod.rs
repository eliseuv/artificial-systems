use std::{
    fmt::Display,
    time::{Duration, Instant},
};

use rand_xoshiro::Xoshiro256PlusPlus;

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

/// Default PRNG to be used
pub type DefaultRNG = Xoshiro256PlusPlus;

/// Simple timer
#[derive(Debug)]
pub struct Timer {
    name: String,
    start: Instant,
}

impl Timer {
    /// Create new timer
    #[inline(always)]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
        }
    }

    /// Start the timer
    #[inline(always)]
    pub fn start(&mut self) {
        self.start = Instant::now();
    }

    /// Read the elapsed time
    #[inline(always)]
    pub fn read(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Display for Timer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[Timer] {name}: {seconds}s",
            name = self.name,
            seconds = self.read().as_secs_f64()
        )
    }
}

/// Automatic Timer
/// Starts timing when it is created and prints the elapsed time to stdout when it is dropped.
#[derive(Debug)]
pub struct AutoTimer(Timer);

impl AutoTimer {
    // Create a timer with a given name
    #[inline(always)]
    pub fn new(name: &str) -> Self {
        Self(Timer::new(name))
    }
}

impl Drop for AutoTimer {
    // Print elapsed time to stdout when dropped
    #[inline(always)]
    fn drop(&mut self) {
        println!("\n{timer}", timer = self.0)
    }
}
