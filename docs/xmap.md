# What does `edm xmap a b` do?

<script src="../assets/manifold.js" defer></script>
<script src="../assets/xmap.js" defer></script>

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

The library set is (by default) the first $L$ points of $M_a$.
The library size parameter $L$ is set by the Stata parameter `library`.

!!! tip "Choose a value for $L$"
    <div class="slider-container"><input type="range" min="0" max="10" value="3" class="slider" id="library"></div>

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = ${L} \]" />

On the other hand, the prediction set will include every point of the $a$ embedding so:

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = M_a = ${P} \]" />

The $b$ time series will be the values which we try to predict.
Here we are trying to predict $p$ observations ahead, where the default case is actually $p = 0$.
The $p = 0$ case means we are using the $a$ time series to try to predict the contemporaneous value of $b$.
A negative $p$ may be chosen, though this is a bit abnormal.

!!! tip "Choose a value for $p$"
    <div class="slider-container"><input type="range" min="-5" max="5" value="0" class="slider" id="p"></div>

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = ${L} \quad \underset{\small \text{Matches}}{\Rightarrow}  y^{\,\mathscr{L}} = ${y_L} \]" />

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = ${P} \quad \underset{\small \text{Matches}}{\Rightarrow}  y^{\,\mathscr{P}} = ${y_P} \]" />

The prediction procedure is then the same as previous times, though the library and prediction sets all contain values from the $a$ time series whereas the $y$ projection vectors contain (usually contemporaneous) values from the $b$ time series.

\[
    \begin{aligned}
        \underbrace{ \text{For target }y_i^{\mathscr{P}} }_{\text{Based on } b}
        & \underset{\small \text{Get predictee}}{\Rightarrow}
        \underbrace{ \mathscr{P}_{i} }_{\text{Based on } a}
        \underset{\small \text{Find neighbours in}}{\Rightarrow}
        \underbrace{ \mathscr{L} }_{\text{Based on } a} \\
        &\,\,\,\,\underset{\small \text{Matches}}{\Rightarrow}
        \underbrace{ \{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)} }_{\text{Based on } b}
        \underset{\small \text{Make prediction}}{\Rightarrow}
        \underbrace{ \hat{y}_i^{\mathscr{P}} }_{\text{Based on } b}
    \end{aligned}
\]
