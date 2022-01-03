# Adding extra variables to the manifold

<script src="../assets/manifold.js" defer></script>
<script src="../assets/adding-to-manifold.js" defer></script>

It can be advantageous to combine data from multiple sources into a single EDM analysis.

The `extra` command will incorporate additional pieces of data into the manifold.
As an example, consider the Stata command

`edm explore x, extra(y)`

???+ tip "Choose the number of observations"
    <div class="slider-container"><input type="range" min="1" max="20" value="10" class="slider" id="numObs"></div>

???+ tip "Choose a value for $E$"
    <div class="slider-container"><input type="range" min="1" max="10" value="2" class="slider" id="E"></div>

???+ tip "Choose a value for $\tau$"
    <div class="slider-container"><input type="range" min="1" max="5" value="1" class="slider" id="tau"></div>

The time-delayed embedding of the $x$ time series with the given $E$ and $\tau$ is the manifold:

<span class="dynamic-equation" data-equation="\[ M_x := \text{Manifold}(x, E,\tau) = ${M_x} \]" />

<!-- 
# ╔═╡ 982d9067-da5b-4cbf-bcba-5c94ed2479b9
begin
	function manifold_with_extra(x, E, tau, extra)
		Mrows = [reshape(x[(i + tau*(E-1)):-tau:(i)], 1, E) for i = 1:(obs-(E-1)*tau)]
		M_x = reduce(vcat, Mrows)
		extraCol = [extra[(i + tau*(E-1))] for i = 1:(obs-(E-1)*tau)]
		hcat(M_x, extraCol)
	end;
	M_x_extra = manifold_with_extra(x, E, τ, y);
	#M_x_extra_set = manifold_set(M_x_extra);
	#L"M_{x,y} := %$M_x_extra_set"
	L"M_{x,y} := %$(latexify(M_x_extra))"
end -->

After extra variables are added, the manifold $M_{x,y}$ no longer has $E$ columns.
In these cases, we make a distinction between $E$ which selects the number of lags for each time series, and the *actual* $E$ which is size of each point (i.e. the number of columns).

By default just one $y$ observation is added to each point in the manifold.

If $E$ lags of $y$ are required, then the command should be altered slightly to

`edm explore x, extra(y(e))`

and then the manifold will be:

<!-- begin
	M_extras = manifold(y, E, τ);
	M_x_extras = hcat(M_x, M_extras)	
	#M_x_extras_set = manifold_set(M_x_extras)
	#L"M_{x,y} := %$M_x_extras_set"
	L"M_{x,y} := %$(latexify(M_x_extras))" 
end -->

More than one `extra` variable can be added.

!!! note
    If some extras are lagged extra variables are specified after some unlagged extras, then the package will reorder them so that all the lagged extras are first.
