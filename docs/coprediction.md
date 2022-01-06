# Coprediction

<script src="../assets/manifold.js" defer></script>
<!-- <script src="../assets/coprediction.js" defer></script> -->

## What does `copredict` do in explore mode?

Imagine that we use the command:

``` stata
edm explore x, copredictvar(z) copredict(out)
```

This will first do a normal

``` stata
edm explore x
```

operation, then it will perform a second set of *copredictions*.
This brings in a second time series $z$, and specifies that the predictions made in copredict mode should be stored in the Stata variable named `out`.

In coprediction mode, the training set will include the entirety of the $M_x$ manifold and its projections:

## What does `copredict` do in xmap mode?

Imagine that we use the command:

``` stata
edm xmap u v, oneway copredictvar(w) copredict(out)
```

Now we combine three different time series to create the predictions in the `out` Stata variable.


!!! tip "Choose the number of observations"
    <div class="slider-container"><input type="range" min="1" max="20" value="10" class="slider" id="numObs"></div>

!!! tip "Choose a value for $E$"
    <div class="slider-container"><input type="range" min="1" max="10" value="2" class="slider" id="E"></div>

!!! tip "Choose a value for $\tau$"
    <div class="slider-container"><input type="range" min="1" max="5" value="1" class="slider" id="tau"></div>

<!-- begin
	u = [symbols("u_$i") for i in 1:obs]
	v = [symbols("v_$i") for i in 1:obs]

	M_u = manifold(u, E, τ);
	P_xmap = M_u

	v_fut = [symbols("v_$(i + τ*(E-1) + p_xmap)") for i = 1:(obs-(E-1)*τ)]
end; -->

In this case, the training set contains all the points in $M_u$:

<!-- begin
	matchStr = raw"\underset{\small \text{Matches}}{\Rightarrow} "

	M_u_str = (latexify(P_xmap))

 	y_L_str_xmap_copred = latexify(v_fut, env=:raw)

	L"\mathscr{L} = M_u = %$M_u_str \quad %$matchStr \quad y^{\,\mathscr{L}} = %$y_L_str_xmap_copred"
end -->

<!-- begin
	M_x_str
	matchStr
	y_L_copred_str = latexify(x_fut, env=:raw)
L"\mathscr{L} = M_x = %$M_x_str \quad %$matchStr \quad y^{\,\mathscr{L}} = %$y_L_copred_str"
end -->

The main change in coprediction is the prediction set and the targets are based on the new $w$ time series:

<!-- begin
	w = [symbols("w_$i") for i in 1:obs]
	M_w = manifold(w, E, τ);
	P_xmap_copred = M_w

	matchStr

	co_ahead = p_xmap
	w_fut = [symbols("w_$(i + τ*(E-1) + co_ahead)") for i = 1:(obs-(E-1)*τ)]
	y_P_str_xmap_copred = latexify(w_fut, env=:raw)


	L"\mathscr{P} = M_w = %$(latexify(P_xmap_copred, env=:raw)) \quad %$matchStr \quad y^{\mathscr{P}} = %$y_P_str_xmap_copred"
end -->

Finally, the simplex prediction steps are the same, with:

\[ 	\underbrace{ \text{For target }y_i^{\mathscr{P}} }_{\text{Based on } w}
	\underset{\small \text{Get predictee}}{\Rightarrow}
	\underbrace{ \mathscr{P}_{i} }_{ \text{Based on } w }
	\underset{\small \text{Find neighbours in}}{\Rightarrow}
	\underbrace{ \mathscr{L} }_{\text{Based on } u}
	\underset{\small \text{Matches}}{\Rightarrow}
	\underbrace{ \{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)} }_{\text{Based on } v}
	\underset{\small \text{Make prediction}}{\Rightarrow}
	\underbrace{ \hat{y}_i^{\mathscr{P}} }_{\text{Based on } v} \]
