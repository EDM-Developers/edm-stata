# The `xmap` subcommand

<script src="../assets/manifold.js" defer></script>
<script src="../assets/xmap.js" defer></script>

The `explore` subcommand focuses on using a the history of a time series to predict itself in the future.

On the other hand, the `xmap` subcommand - which is short for _cross mapping_ - focuses on using one time series to predict the value of a different time series.

This process is central to the 'convergent cross mapping' algorithm which we use to decide if one time series has a causal effect on another.

## Setup

Imagine that we use the command:

``` stata
edm xmap a b, oneway
```

This will consider two different time series, here labelled $a$ and $b$.

!!! tip "Choose the number of observations"
    <div class="slider-container"><input type="range" min="1" max="20" value="10" class="slider" id="numObs"></div>

In tabular form, the data looks like:

<span class="dynamic-equation" data-equation="\[ a = ${a_time_series} \quad \text{and} \quad b = ${b_time_series} \]" />

!!! tip "Choose a value for $E$"
    <div class="slider-container"><input type="range" min="1" max="10" value="2" class="slider" id="E"></div>

!!! tip "Choose a value for $\tau$"
    <div class="slider-container"><input type="range" min="1" max="5" value="1" class="slider" id="tau"></div>

The lagged embedding $M_a$ is constructed:

<span class="dynamic-equation" data-equation="\[ M_a = ${M_a} \]" />

## Library and prediction sets

In `xmap` mode, the library set $\mathscr{L}$ is typically the first $L$ points of $M_a$.
The library size parameter $L$ is set by the Stata parameter `library`.

!!! tip "Choose a value for $L$"
    <div class="slider-container"><input type="range" min="0" max="10" value="3" class="slider" id="library"></div>

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = ${L} \]" />

However, in `xmap` mode the prediction set $\mathscr{P}$ will include every point of the $a$ embedding so:

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = M_a = ${P} \]" />

## Targets come from the other time series

The $b$ time series will be the values which we try to predict.
Here we are trying to predict $p$ observations ahead, where the default case is actually $p = 0$.
The $p = 0$ case means we are using the $a$ time series to try to predict the contemporaneous value of $b$.
A negative $p$ may be chosen, though this is a bit abnormal.

!!! tip "Choose a value for $p$"
    <div class="slider-container"><input type="range" min="-5" max="5" value="0" class="slider" id="p"></div>

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = ${L} \quad \underset{\small \text{Matches}}{\Rightarrow}  \mathbf{y}_{\mathscr{L}} = ${y_L} \]" />

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = ${P} \quad \underset{\small \text{Matches}}{\Rightarrow}  \mathbf{y}_{\mathscr{P}} = ${y_P} \]" />

## What does `edm xmap a b` do?

In `xmap`, we perform a series of predictions in very similar manner to the `explore` subcommand. 
The difference is our library and prediction sets contain values from the $a$ time series whereas the $\mathbf{y}_{\mathscr{L}}$ and $\mathbf{y}_{\mathscr{P}}$ vectors contain (usually contemporaneous) values from the $b$ time series.

\[
    \begin{aligned}
        \text{Given } \underbrace{ \mathbf{x}_{i} }_{\text{From } a} \text{ and target }\underbrace{ y_i }_{\text{From } b}
        & 
        \underset{\small \text{Find neighbours in } \mathscr{L} }{\Rightarrow}
        ( \underbrace{ \mathbf{x}_{[j]} }_{\text{From } a} , \underbrace{ y_{[j]} }_{\text{From } b} )_{j \in \mathcal{NN}_k(i)}
        \underset{\small \text{Make prediction}}{\Rightarrow}
        \underbrace{ \hat{y}_i }_{\text{For } b }
    \end{aligned}
\]

This means that we are learning the mapping from (the recent history of) the $a_t$ time series to the (contemporaneous value of the) $b_t$ time series.

## Convergent cross mapping

Typically the goal of using `xmap` is to specify a grid of values for the `library` size and look at the relationship between the library size $L$ and the predictive performance $\rho$.

If the `xmap a b` predictive performance increases significantly when more data to learn from (i.e. as $L$ increases) then the convergent cross mapping (CCM) algorithm says that this is evidence that the time series $b_t$ has a causal effect on $a_t$.

This may seem backwards at first glance, but the logic is that if $b_t$ has a causal effect on $a_t$, then information about $b_t$ is embedded in the time series $a_t$.

Hence, if we get more and more data from $a_t$, we can glean more information about the cause $b_t$ and hence make better prediction of $b_t$ as $L$ increases.

The summary which the `edm` command prints out uses the notation `b|M(a)` to indicate that we use the reconstructed manifold of $a_t$ to make predictions about $b_t$, and so the causal effect of $b$ to $a$ can be read left-to-right.

See the examples/vignettes to see the CCM process in action.