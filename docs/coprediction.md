# Coprediction

<script src="../assets/manifold.js" defer></script>
<script src="../assets/coprediction.js" defer></script>

## What does `copredict` do in explore mode?

Imagine that we use the command:

``` stata
edm explore a, copredictvar(c) copredict(out)
```

This will first do a normal

``` stata
edm explore a
```

operation, then it will perform a second set of *copredictions*.
This brings in a second time series $c$, and specifies that the predictions made in copredict mode should be stored in the Stata variable named `out`.

For the following, let's first set the general manifold parameters.

!!! tip "Choose the number of observations"
    <div class="slider-container"><input type="range" min="1" max="20" value="10" class="slider" id="numObs"></div>

!!! tip "Choose a value for $E$"
    <div class="slider-container"><input type="range" min="1" max="10" value="2" class="slider" id="E"></div>

!!! tip "Choose a value for $\tau$"
    <div class="slider-container"><input type="range" min="1" max="5" value="1" class="slider" id="tau"></div>

In coprediction mode, the training set will include the entirety of the $M_a$ manifold and its projections:

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = M_a = ${M_a} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad \mathbf{y}_{\mathscr{L}}  = ${y_L_a} \]" />

In copredict mode the most significant difference is that we change $\mathscr{P}$ to be the $M_c$ manifold for the $c$ time series and $\mathbf{y}_{\mathscr{P}} $ to:

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = M_c = ${M_c} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad \mathbf{y}_{\mathscr{P}}  = ${y_P_c} \]" />

The rest of the simplex procedure is the same as before:

\[
    \begin{aligned}
        \underbrace{ \text{For target } y_i }_{ \text{Based on } c }
        & \underset{\small \text{Get predictee}}{\Rightarrow}
        \underbrace{ \mathbf{x}_{i} }_{ \text{Based on } c}
        \underset{\small \text{Find neighbours in } \mathscr{L}}{\Rightarrow}
        \mathcal{NN}_k(i) \\
        &\,\,\,\,
        \underset{\small \text{Extracts}}{\Rightarrow}
        \{ y_{[j]}\}_{j \in \mathcal{NN}_k(i)}
        \underset{\small \text{Make prediction}}{\Rightarrow}
        \hat{y}_i
    \end{aligned}
\]

## What does `copredict` do in xmap mode?

Imagine that we use the command:

``` stata
edm xmap a b, oneway copredictvar(c) copredict(out)
```

Now we combine three different time series to create the predictions in the `out` Stata variable.

In this case, the training set contains all the points in $M_a$:

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = M_a = ${M_a_xmap} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad \mathbf{y}_{\mathscr{L}}  = ${y_L_b_xmap} \]" />

The main change in coprediction is the prediction set and the targets are based on the new $c$ time series:

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = M_c = ${M_c_xmap} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad \mathbf{y}_{\mathscr{P}}  = ${y_P_c_xmap} \]" />


Finally, the simplex prediction steps are the same, with:

\[ 	\underbrace{ \text{For target }y_i }_{\text{Based on } c}
	\underset{\small \text{Get predictee}}{\Rightarrow}
	\underbrace{ \mathbf{x}_i }_{ \text{Based on } c }
	\underset{\small \text{Find neighbours in}}{\Rightarrow}
	\underbrace{ \mathscr{L} }_{\text{Based on } a}
	\underset{\small \text{Matches}}{\Rightarrow}
	\underbrace{ \{ y_{[j]}\}_{j \in \mathcal{NN}_k(i)} }_{\text{Based on } b}
	\underset{\small \text{Make prediction}}{\Rightarrow}
	\underbrace{ \hat{y}_i }_{\text{Based on } b} \]
