# Coprediction

<script src="../assets/manifold.js" defer></script>
<script src="../assets/coprediction.js" defer></script>

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

For the following, let's first set the general manifold parameters.

!!! tip "Choose the number of observations"
    <div class="slider-container"><input type="range" min="1" max="20" value="10" class="slider" id="numObs"></div>

!!! tip "Choose a value for $E$"
    <div class="slider-container"><input type="range" min="1" max="10" value="2" class="slider" id="E"></div>

!!! tip "Choose a value for $\tau$"
    <div class="slider-container"><input type="range" min="1" max="5" value="1" class="slider" id="tau"></div>

In coprediction mode, the training set will include the entirety of the $M_x$ manifold and its projections:

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = M_x = ${M_x} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y^{\,\mathscr{L}} = ${y_L_x} \]" />

In copredict mode the most significant difference is that we change $\mathscr{P}$ to be the $M_z$ manifold for the $z$ time series and $y^{\mathscr{P}}$ to:

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = M_z = ${M_z} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y^{\mathscr{P}} = ${y_P_z} \]" />

The rest of the simplex procedure is the same as before:

\[
    \begin{aligned}
        \underbrace{ \text{For target }y_i^{\mathscr{P}} }_{ \text{Based on } z }
        & \underset{\small \text{Get predictee}}{\Rightarrow}
        \underbrace{ \mathscr{P}_{i} }{ \text{Based on } z}
        \underset{\small \text{Find neighbours in } \mathscr{L}}{\Rightarrow}
        \mathcal{NN}_k(i) \\
        &\,\,\,\,
        \underset{\small \text{Extracts}}{\Rightarrow}
        \{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)}
        \underset{\small \text{Make prediction}}{\Rightarrow}
        \hat{y}_i^{\mathscr{P}}
    \end{aligned}
\]

## What does `copredict` do in xmap mode?

Imagine that we use the command:

``` stata
edm xmap u v, oneway copredictvar(w) copredict(out)
```

Now we combine three different time series to create the predictions in the `out` Stata variable.

In this case, the training set contains all the points in $M_u$:

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = M_u = ${M_u} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y^{\,\mathscr{L}} = ${y_L_v} \]" />

The main change in coprediction is the prediction set and the targets are based on the new $w$ time series:

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = M_w = ${M_w} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y^{\mathscr{P}} = ${y_P_w} \]" />


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
