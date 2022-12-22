# Adding extra variables to the manifold

<script src="../assets/manifold.js" defer></script>
<script src="../assets/adding-to-manifold.js" defer></script>

It can be advantageous to combine data from multiple sources into a single EDM analysis.

The `extra` command will incorporate additional pieces of data into the manifold.
As an example, consider the Stata command

`edm explore a, extra(b)`

!!! tip "Choose the number of observations"
    <div class="slider-container"><input type="range" min="1" max="20" value="5" class="slider" id="numObs"></div>

!!! tip "Choose a value for $E$"
    <div class="slider-container"><input type="range" min="1" max="10" value="2" class="slider" id="E"></div>

!!! tip "Choose a value for $\tau$"
    <div class="slider-container"><input type="range" min="1" max="5" value="1" class="slider" id="tau"></div>

The time-delayed embedding of the $a$ time series with the given $E$ and $\tau$ is the manifold:

<span class="dynamic-equation" data-equation="\[ M_a = ${M_a} \]" />

However, with the `extra(b)` option, we use the time-delayed embedding with the extra time series included like:

<span class="dynamic-equation" data-equation="\[ M_{a,b} = ${M_a_b} \]" />

After extra variables are added, the manifold $M_{a,b}$ no longer has $E$ columns.
In these cases, we make a distinction between $E$ which selects the number of lags for each time series, and the *actual* $E$ which is size of each point (i.e. the number of columns).

By default just one $b$ observation is added to each point in the manifold.
However, this will only allow us to capture to the contemporaneous effect of $b$ on the system.

If instead we want to add the last $E$ lags of $b$ (just as we do with the $a$ time series series), then the command should be altered slightly to

`edm explore a, extra(b(e))`

and then the manifold will be:

<span class="dynamic-equation" data-equation="\[ M_{a,b} = ${M_a_b_varying} \]" />

More than one `extra` variable can be added, and any combinations of $E$-varying and non-$E$-varying extras are permitted.

!!! note
    If some extras are lagged extra variables (i.e. $E$-varying extras) and they are specified after some unlagged extras, then the package will reorder them so that all the lagged extras are first.
