/// A macro to calculate hypercube coordinates without runtime loops.
///
/// It takes the linear index, the side length, and a list of variable names
/// to bind the results to.
///
/// # Usage
/// `hypercube_index!(index, side_length; x, y, z)`
///
/// # Logic
/// It unrolls the division/modulo operations recursively. It processes the
/// tail of the list first to ensure Row-Major ordering (where the last
/// variable changes the fastest).
macro_rules! hypercube_index {
    // Entry point: Wraps everything in a block to keep scope clean
    ($index:expr, $side:expr; $($vars:ident),+) => {
        {
            let mut _idx = $index;
            let _side = $side;

            // Invoke the internal recursive rules
            hypercube_index!(@recurse _idx, _side, $($vars),+);

            // Return the array of variables defined by the recursion
            [$($vars),+]
        }
    };

    // Internal Rule 1: Base case (Last single variable)
    // This matches when there is only one variable left (e.g., 'z').
    // In Row-Major, this is the fastest changing dimension (calculated first).
    (@recurse $idx:ident, $side:ident, $last:ident) => {
        let $last = $idx % $side;
        $idx /= $side;
    };

    // Internal Rule 2: Recursive step (Head + Tail)
    // Matches 'x' as head and 'y, z' as tail.
    // Crucially, it calls itself on the TAIL first. This ensures 'z' is calculated
    // before 'y', and 'y' before 'x'.
    (@recurse $idx:ident, $side:ident, $head:ident, $($tail:ident),+) => {
        // Recurse deeper first (Process the end of the list)
        hypercube_index!(@recurse $idx, $side, $($tail),+);

        // Process the current head after the tail has consumed the lower bits
        let $head = $idx % $side;
        $idx /= $side;
    };
}

// Re-export macro
pub(crate) use hypercube_index;
