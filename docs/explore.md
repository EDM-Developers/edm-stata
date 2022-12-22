# What does `edm explore a` do?

<script src="../assets/manifold.js" defer></script>
<script src="../assets/explore.js" defer></script>

## First split into library and prediction sets

Firstly, the manifold $M_a$ is split into two parts, called the *library set* denoted $\mathscr{L}$ and the *prediction set* denoted $\mathscr{P}$.
By default, we take the points of the $M_a$ manifold and assign the first half of them to $\mathscr{L}$ and the second half to $\mathscr{P}$.

!!! note
    In the default case, the same point doesn't appear in both $\mathscr{L}$ and $\mathscr{P}$, though given other options then the same point may appear in both sets.

Starting with the time-delayed embedding of $a$.

!!! tip "Choose the number of observations"
    <div class="slider-container"><input type="range" min="1" max="20" value="10" class="slider" id="numObs"></div>

!!! tip "Choose a value for $E$"
    <div class="slider-container"><input type="range" min="1" max="10" value="2" class="slider" id="E"></div>

!!! tip "Choose a value for $\tau$"
    <div class="slider-container"><input type="range" min="1" max="5" value="1" class="slider" id="tau"></div>

The time-delayed embedding of the $a$ time series with the selected $E$ and $\tau$ is the manifold:

<span class="dynamic-equation" data-equation="\[ M_a = ${M_a} \]" />

Then we take the first half of the points to create the library set, which leaves the remaining points to create the prediction set.

In that case, the *library set* is

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = ${L} \]" />

and the *prediction set* is

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = ${P} \]" />

It will help to introduce a notation to refer to a specific point in these sets based on its row number.
E.g. in the example above, the first point in the library set takes the value:

<span class="dynamic-equation" data-equation="\[ \mathbf{x}_{[1]} = ${L_1} \]" />

More generally $\mathbf{x}_{[i]}$ refers to the $i$th point in $\mathscr{L}$
while $\mathbf{x}_{j}$ refers to the $j$th point in $\mathscr{P}$.

## Next, look at the future values of each point

Each point on the manifold refers to a small trajectory of a time series, and for each point we look $p$ observations into the future of the time series.

!!! tip "Choose a value for $p$"
    <div class="slider-container"><input type="range" min="-5" max="5" value="1" class="slider" id="p"></div>

So if we take the first point of the prediction set $\mathbf{x}_{1}$ and say that $y_1$ is the value it takes $p$ observations in the future, we get:

<span class="dynamic-equation" data-equation="\[\mathbf{x}_{[1]} = ${L_1} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y_{[1]}  = ${y_L_1} \]" />

<span class="dynamic-equation" data-equation="\[\mathbf{x}_{1} = ${P_1} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y_1  = ${y_P_1} \]" />


This $p$ may be thought of as the *prediction horizon*, and in `explore` mode is defaults to $\tau$ and in `xmap` mode it defaults to 0.

!!! note "Our $p$ versus the $T_p$ which is common in the literature"
    In the literature, instead of measuring the number of observations $p$ ahead, authors normally use the value $T_p$ to denote the amount of time this corresponds to.
    When data is regularly sampled (e.g. $t_i = i$) then there is no difference (e.g. $T_p = p$), however for irregularly sampled data the actual time difference may be different for each prediction.

In the training set, this means each point of $\mathscr{L}$ matches the corresponding value in $\mathbf{y}_{\mathscr{L}}$:

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = ${L} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad \mathbf{y}_{\mathscr{L}} = ${y_L} \]" />

Similarly, for the prediction set:

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = ${P} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad \mathbf{y}_{\mathscr{P}} = ${y_P} \]" />

We may refer to elements of the $y^{\mathscr{L}}$ vector as *projections* as they come about by taking the $a$ time series and projecting it into the future by $p$ observations.

## What does `edm explore a` predict?

When running `edm explore a`, we pretend that we don't know the values in $\mathbf{y}_{\mathscr{P}}$ and that we want to predict them given we know $\mathscr{L}$ and $\mathbf{y}_{\mathscr{L}}$.

The first prediction is to try to find the value of $y_1$ given the corresponding point $\mathbf{x}_1$:

<span class="dynamic-equation" data-equation="\[\mathbf{x}_{1} = ${P_1} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y_1  = \, ??? \]" />

We will use $\mathbf{x}_1$ to predict $y_1$, so the $\mathbf{x}_1$ values can be viewed as *covariates* and $y_1$ as the *target* of the prediction.

### Using the simplex algorithm

If we have chosen the `algorithm` to be the simplex algorithm, then we start by finding the $k$ points in $\mathscr{L}$ which are closest to the given $\mathbf{x}_1$.


Let's say that we chose $k=2$ and also pretend that the two most similar points to $\mathbf{x}_1$ are $\mathbf{x}_{[3]}$ and $\mathbf{x}_{[5]}$.
We will choose the notation 

\[
    \mathcal{NN}_k(\mathbf{x}_{1}) = \{ 3, 5 \}
\]

to describe the set of the indices for the $k$ nearest neighbours of $\mathbf{x}_{1}$.

In this process, we also store the distances (by default, using Euclidean distance) between $\mathbf{x}_{1}$ and its nearest neighbours, giving special attention to the smallest distance $d_{\text{min}} > 0$.

Then, we predict that

\[
    \hat{y}_1 := w_1 \times y_{[3]} + w_2 \times y_{[5]}
\]

where $w_1$ and $w_2$ are some weights with add up to 1. Basically we are predicting that $y_1$ is a weighted average of the $y_{[j]}$ points for each $j \in \mathcal{NN}_k(\mathbf{x}_{1})$.

Specifically, the weights depend upon the distance from $\mathbf{x}_{1}$ to the corresponding points in the library set.

If weight $w_i$ corresponds to library point $\mathbf{x}_{[j]}$ then we have

\[
    w_i \propto \exp\bigl\{ -\theta \, d( \mathbf{x}_{i} , \mathbf{x}_{[j]} ) / d_{\text{min}} \bigr\} \,.
\]

Here, the notation $d(\mathbf{x}_i, \mathbf{x}_{[j]})$ refers to the distance between prediction point $i$ and library point $j$, and $\theta \ge 0$ is a hyperparameter chosen by the user.

To summarise the process in a rough algorithm, we:

1. Loop through the prediction set, letting $i = 1$ to $| \mathscr{P} |$:
    1. Start with our covariates and target from the prediction set $\mathbf{x}_{i}$ and $y_i$, though we temporarily pretend that we don't know $y_i$.
    2. We find points in the library set that are close to $\mathbf{x}_{i}$; in effect, this is constructing a very localised training set of pairs $\mathbf{x}_{[j]} \to y_{[j]}$ for $j \in \mathcal{NN}_k(\mathbf{x}_{i})$.
    3. Make the prediction $\hat{y}_i$ as a weighted average of the $y_{[j]}$ neighbour targets.
4. Summarise the prediction accuracy (e.g. the $\rho$ correlation) between the $\hat{y}_i$ predictions and the true $y_i$ values.


### Using the S-map algorithm

The alternative choice for `algorithm` is the S-map procedure, which is short for *Sequential Locally Weighted Global Linear Maps*.

This prediction algorithm uses linear regression to predict each point in the prediction set, though importantly it uses a **different** linear model for each prediction.

#### Using linear regression

To make the contrast obvious, let's imagine how traditional linear regression could be used to take some covariates $\mathbf{x}_{i}$ from $\mathscr{P}$ and try to predict $y_i$.

Linear regression would take the library set and solve the linear system $\overline{\mathscr{L}} \boldsymbol{\beta} = \mathbf{y}_{\mathscr{L}}$ for the $\boldsymbol{\beta} \in \mathbb{R}^{E+1}$ coefficients given $\overline{\mathscr{L}} = [\mathbf{1}; \mathscr{L}]$ (similarly $\overline{\mathbf{x}}_i$ is $\mathbf{x}_i$ prepended by a 1).

These coeffecients could then be used to make predictions like

\[
    \hat{y}_i = \overline{\mathbf{x}}_{i}^\top \boldsymbol{\beta}
\]

As the $\overline{\mathbf{x}}_i$'s are short time series trajectories, this method of forecasting is equivalent to fitting an autoregressive process of length $E$.

#### Back to S-map

The S-map procedure deviates from this process by creating a customised $\boldsymbol{\beta}_i$ vector when making each prediction given $\mathbf{x}_i$ rather than using a shared $\boldsymbol{\beta}$ for all of the system.

It achieves this by replacing the traditional regression loss function of 

\[
    \hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} | \mathbf{y}_{\mathscr{L}} - \overline{\mathscr{L}} \boldsymbol{\beta} |^2
\]

with a weighted form

\[
    \hat{\boldsymbol{\beta}}_i = \arg\min_{\boldsymbol{\beta}} \bigl| \mathbf{w}_i \circ \bigl( \mathbf{y}_{\mathscr{L}} - \overline{\mathscr{L}} \boldsymbol{\beta} \bigr) \bigr|^2 \,.
\]

The S-map loss function has the extra $\mathbf{w}_i$ weight vector (included by elementwise multiplication) and the $j$ element of the vector is given by

\[
    w_{i,j} = \exp\bigl\{ -\theta \, d( \mathbf{x}_{i} , \mathbf{x}_{[j]} ) / d_{\text{mean}} \bigr\} \,.
\]

In this equation, $d_{\text{mean}}$ refers to the average distance between $\mathbf{x}_{i}$ and every point in $\mathscr{L}$.


Given the specific

<span class="dynamic-equation" data-equation="\[ \mathbf{x}_{1} = ${P_1} \]" />

in this example, the prediction would look like:

<span class="dynamic-equation" data-equation="\[ y_{1} \approx \hat{y}_1 := ${yhat_P_1} \]" />

To summarise the whole S-map procedure:

\[
    \begin{aligned}
        \text{For target }y_i
        & \underset{\small \text{Get predictee}}{\Rightarrow}
        \mathbf{x}_{i}
        \underset{\small \text{Find neighbours in } \mathscr{L}}{\Rightarrow}
        \mathcal{NN}_k(i) \\
        &\,\,\,\,
        \underset{\small \text{Extracts}}{\Rightarrow}
        \{ \mathbf{x}_{[j]}, y_j^{\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)}
        \underset{\small \text{Calculate}}{\Rightarrow}
        \{ w_{i,j} \}_{j=1,\ldots,E}
        \underset{\small \text{Make prediction}}{\Rightarrow}
        \hat{y}_i
    \end{aligned}
\]

## Assessing the prediction quality

We calculate the $\hat{y}_i$ predictions for each target in the prediction set (so $i = 1, \dots, |\mathscr{P}|$), and store the predictions in a vector $\hat{y}$.

As we observe the true value of $y_i$ for most (if not all) of the targets in the prediction set, we can compare our $\hat{y}_i$ predictions to the observed values.
We assess the quality of the predictions using either the correlation

\[ \rho := \text{Correlation}(y , \hat{y}) \]

or using the mean absolute error

\[ \text{MAE} := \frac{1}{| \mathscr{P} |} \sum_{i=1}^{| \mathscr{P} |} | y_i - \hat{y}_i | . \]
