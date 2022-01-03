# What does `edm xmap u v` do?

<script src="../assets/manifold.js" defer></script>
<script src="../assets/xmap.js" defer></script>

Imagine that we use the command:

``` stata
edm xmap u v, oneway
```

This will consider two different time series, here labelled $u$ and $v$.

!!! tip "Choose the number of observations"
    <div class="slider-container"><input type="range" min="1" max="20" value="10" class="slider" id="numObs"></div>

!!! tip "Choose a value for $E$"
    <div class="slider-container"><input type="range" min="1" max="10" value="2" class="slider" id="E"></div>

!!! tip "Choose a value for $\tau$"
    <div class="slider-container"><input type="range" min="1" max="5" value="1" class="slider" id="tau"></div>

The lagged embedding $M_u$ is constructed:

<span class="dynamic-equation" data-equation="\[ u = ${u_time_series} \Rightarrow M_u = ${M_u} \]" />

The library set is (by default) the first $L$ points of $M_u$.
The library size parameter $L$ is set by the Stata parameter `library`.

!!! tip "Choose a value for $L$"
    <div class="slider-container"><input type="range" min="3" max="10" value="3" class="slider" id="library"></div>

<!-- Technically, max of this slider should be size(M_u, 1) -->

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = ${L} \]" />

On the other hand, the prediction set will include every point of the $u$ embedding so:

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = M_u = ${P} \]" />

The $v$ time series will be the values which we try to predict.
Here we are trying to predict $p$ observations ahead, where the default case is actually $p = 0$.
The $p = 0$ case means we are using the $u$ time series to try to predict the contemporaneous value of $v$.
A negative $p$ may be chosen, though this is a bit abnormal.

!!! tip "Choose a value for $p$"
    <div class="slider-container"><input type="range" min="-5" max="5" value="0" class="slider" id="p"></div>

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = ${L} \quad \underset{\small \text{Matches}}{\Rightarrow}  y^{\,\mathscr{L}} = ${y_L} \]" />

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = ${P} \quad \underset{\small \text{Matches}}{\Rightarrow}  y^{\,\mathscr{P}} = ${y_P} \]" />

The prediction procedure is then the same as previous times, though the library and prediction sets all contain values from the $u$ time series whereas the $y$ projection vectors contain (usually contemporaneous) values from the $v$ time series.

\[
    \begin{aligned}
        \underbrace{ \text{For target }y_i^{\mathscr{P}} }_{\text{Based on } v}
        & \underset{\small \text{Get predictee}}{\Rightarrow}
        \underbrace{ \mathscr{P}_{i} }_{\text{Based on } u}
        \underset{\small \text{Find neighbours in}}{\Rightarrow}
        \underbrace{ \mathscr{L} }_{\text{Based on } u} \\
        &\,\,\,\,\underset{\small \text{Matches}}{\Rightarrow}
        \underbrace{ \{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)} }_{\text{Based on } v}
        \underset{\small \text{Make prediction}}{\Rightarrow}
        \underbrace{ \hat{y}_i^{\mathscr{P}} }_{\text{Based on } v}
    \end{aligned}
\]
