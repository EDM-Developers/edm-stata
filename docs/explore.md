# What does `edm explore x` do?

<script src="../assets/manifold.js" defer></script>
<script src="../assets/explore.js" defer></script>

## First split into library and prediction sets

Firstly, the manifold $M_x$ is split into two parts, called the *library set* denoted $\mathscr{L}$ and the *prediction set* denoted $\mathscr{P}$.
By default, we take the points of the $M_x$ manifold and assign the first half of them to $\mathscr{L}$ and the second half to $\mathscr{P}$.

!!! note
    In the default case, the same point doesn't appear in both $\mathscr{L}$ and $\mathscr{P}$, though given other options then the same point may appear in both sets.

Starting with the time-delayed embedding of $x$.

!!! tip "Choose the number of observations"
    <div class="slider-container"><input type="range" min="1" max="20" value="10" class="slider" id="numObs"></div>

!!! tip "Choose a value for $E$"
    <div class="slider-container"><input type="range" min="1" max="10" value="2" class="slider" id="E"></div>

!!! tip "Choose a value for $\tau$"
    <div class="slider-container"><input type="range" min="1" max="5" value="1" class="slider" id="tau"></div>

The time-delayed embedding of the $x$ time series with the selected $E$ and $\tau$ is the manifold:

<span class="dynamic-equation" data-equation="\[ M_x = ${M_x} \]" />

Then we may take the first half of the points to create the library set, which leaves the remaining points to create the prediction set.

In that case, the *library set* is

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = ${L} \]" />

and the *prediction set* is

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = ${P} \]" />

It will help to introduce a notation to refer to a specific point in these sets based on its row number.
E.g. in the example above, the first point in the library set takes the value:

<span class="dynamic-equation" data-equation="\[ \mathscr{L}_1 = ${L_1} \]" />

More generally $\mathscr{L}_{i}$ refers to the $i$th point in $\mathscr{L}$
and similarly $\mathscr{P}_{j}$ refers to the $j$th point in $\mathscr{P}$.

## Next, look at the future values of each point

Each point on the manifold refers to a small trajectory of a time series, and for each point we look $p$ observations into the future of the time series.

!!! tip "Choose a value for $p$"
    <div class="slider-container"><input type="range" min="-5" max="5" value="1" class="slider" id="p"></div>

So if we take the first point of the prediction set $\mathscr{P}_{1}$ and say that $y_1^{\mathscr{P}}$ is the value it takes $p$ observations in the future, we get:

<span class="dynamic-equation" data-equation="\[\mathscr{P}_{1} = ${P_1} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y_1^{\mathscr{P}}  = ${y_P_1} \]" />

This $p$ may be thought of as the *prediction horizon*, and in `explore` mode is defaults to $\tau$ and in `xmap` mode it defaults to 0.

In the literature, instead of measuring the number of observations $p$ ahead, authors normally use the value $T_p$ to denote the amount of time this corresponds to.
When data is regularly sampled (e.g. $t_i = i$) then there is no difference (e.g. $T_p = p$), however for irregularly sampled data the actual time difference may be different for each prediction.

In the training set, this means each point of $\mathscr{L}$ matches the corresponding value in $y^{\,\mathscr{L}}$:

<span class="dynamic-equation" data-equation="\[ \mathscr{L} = ${L} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y^{\,\mathscr{L}} = ${y_L} \]" />

Similarly, for the prediction set:

<span class="dynamic-equation" data-equation="\[ \mathscr{P} = ${P} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y^{\,\mathscr{P}} = ${y_P} \]" />

We may refer to elements of the $y^{\mathscr{L}}$ vector as *projections* as they come about by taking the $x$ time series and projecting it into the future by $p$ observations.

## What does `edm explore x` predict?

When running `edm explore x`, we pretend that we don't know the values in $y^{\mathscr{P}}$ and that we want to predict them given we know $\mathscr{P}$, $\mathscr{L}$ and $y^{\,\mathscr{L}}$.

The first prediction is to try to find the value of $y_1^{\mathscr{P}}$ given the corresponding point $\mathscr{P}_1$:

<span class="dynamic-equation" data-equation="\[\mathscr{P}_{1} = ${P_1} \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y_1^{\mathscr{P}}  = \, ??? \]" />

The terminology we use is that $y_1^{\mathscr{P}}$ is the *target* and the point $\mathscr{P}_1$ is the *predictee*.

Looking over all the points in $\mathscr{L}$, we find the indices of the $k$ points which are the most similar to $\mathscr{P}_{1}$.

Let's pretend we have $k=2$ and the most similar points are $\mathscr{L}_{3}$ and $\mathscr{L}_{5}$.
We will choose the notation $\mathcal{NN}_k(1) = \{ 3, 5 \}$ to describe this set of $k$ nearest neighbours of $\mathscr{P}_{1}$.

### Using the simplex algorithm

Then, if we have chosen the `algorithm` to be the simplex algorithm, we predict that

\[
    y_{1}^{\mathscr{P}} \approx \hat{y}_1^{\mathscr{P}} := w_1 \times y_{3}^{\,\mathscr{L}} + w_2 \times y_{5}^{\,\mathscr{L}}
\]

where $w_1$ and $w_2$ are some weights with add up to 1. Basically we are predicting that $y_1^{\mathscr{P}}$ is a weighted average of the points $\{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(1)}$.

To summarise the whole simplex procedure:

\[
    \begin{aligned}
        \text{For target }y_i^{\mathscr{P}}
        & \underset{\small \text{Get predictee}}{\Rightarrow}
        \mathscr{P}_{i}
        \underset{\small \text{Find neighbours in } \mathscr{L}}{\Rightarrow}
        \mathcal{NN}_k(i) \\
        &\,\,\,\,
        \underset{\small \text{Extracts}}{\Rightarrow}
        \{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)}
        \underset{\small \text{Make prediction}}{\Rightarrow}
        \hat{y}_i^{\mathscr{P}}
    \end{aligned}
\]

### Using the S-map algorithm

If however we chose `algorithm` to be the S-map algorithm, then we predict

\[
    y_{1}^{\mathscr{P}} \approx \hat{y}_1^{\mathscr{P}} := \sum_{j=1}^E w_{1,j} \times  \mathscr{P}_{1j}
\]

where the $\{ w_{1,j} \}_{j=1,\cdots,E}$ weights are calculated by solving a linear system based on the points in $\{ \mathscr{L}_j \}_{j \in \mathcal{NN}_k(1)}$ and $\{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(1)}$.

Given the specific

<span class="dynamic-equation" data-equation="\[ \mathscr{P}_{1} = ${P_1} \]" />

in this example, the prediction would look like:

<span class="dynamic-equation" data-equation="\[ y_{1}^{\mathscr{P}} \approx \hat{y}_1^{\mathscr{P}} := ${yhat_P_1} \]" />

To summarise the whole S-map procedure:

\[
    \begin{aligned}
        \text{For target }y_i^{\mathscr{P}}
        & \underset{\small \text{Get predictee}}{\Rightarrow}
        \mathscr{P}_{i}
        \underset{\small \text{Find neighbours in } \mathscr{L}}{\Rightarrow}
        \mathcal{NN}_k(i) \\
        &\,\,\,\,
        \underset{\small \text{Extracts}}{\Rightarrow}
        \{ \mathscr{L}_j, y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)}
        \underset{\small \text{Calculate}}{\Rightarrow}
        \{ w_{i,j} \}_{j=1,\ldots,E}
        \underset{\small \text{Make prediction}}{\Rightarrow}
        \hat{y}_i^{\mathscr{P}}
    \end{aligned}
\]

## Assessing the prediction quality

We calculate the $\hat{y}_i^{\mathscr{P}}$ predictions for each target in the prediction set (so $i = 1, \dots, |\mathscr{P}|$), and store the predictions in a vector $\hat{y}^{\mathscr{P}}$.

As we have the true value of $y_i^{\mathscr{P}}$ for each target in the prediction set, we can compare our $\hat{y}_i^{\mathscr{P}}$ predictions and assess their quality using their correlation

\[ \rho := \text{Correlation}(y^{\mathscr{P}} , \hat{y}^{\mathscr{P}}) \]

or using the mean absolute error

\[ \text{MAE} := \frac{1}{| \mathscr{P} |} \sum_{i=1}^{| \mathscr{P} |} | y_i^{\mathscr{P}} - \hat{y}_i^{\mathscr{P}} | . \]
