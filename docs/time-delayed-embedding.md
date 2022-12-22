# Time-delayed embedding

<script src="../assets/manifold.js" defer></script>
<script src="../assets/time-delayed-embedding.js" defer></script>

The fundamental logic of empirical dynamic modelling are based on Taken's theorem, and the core idea of Taken's theorem is to create these things called time-delayed embeddings (or time-delayed reconstructions).
Essentially, we take a time series and break it into many overlapping short trajectories of a fixed length.
This page steps through a basic example of this process, highlighting some relevant hyperparameters which can be set.

Imagine that we observe a time series $a$.

!!! tip "Choose the number of observations"
    <div class="slider-container"><input type="range" min="1" max="20" value="10" class="slider" id="numObs"></div>

In tabular form, the data looks like:

<span class="dynamic-equation" data-equation="\[ ${a_time_series} \]" />

So each one of $a_i$ is an *observation* of the $a$ time series.

To create a time-delayed embedding based on any of these time series, we first need to choose the size of the embedding $E$.

!!! tip "Choose a value for $E$"
    <div class="slider-container"><input type="range" min="1" max="10" value="2" class="slider" id="E"></div>

The data may be too finely sampled in time.
So we select a $\tau$ which means we only look at every $\tau$th observation for each time series.

!!! tip "Choose a value for $\tau$"
    <div class="slider-container"><input type="range" min="1" max="5" value="1" class="slider" id="tau"></div>

The time-delayed embedding of the $a$ time series with
<span class="dynamic-inline" data-equation="E = ${E}, \tau = ${tau}" />

is the manifold:

<span class="dynamic-equation" data-equation="\[ M_a := \text{Manifold}(x, E,\tau) = ${M_a_sets} \]" />

The manifold is a collection of these time-delayed *embedding vectors*.
For short, we just refer to each vector as a *point* on the manifold.
While the manifold notation above is the most accurate (a set of vectors) we will henceforward use the more convenient matrix notation:

<span class="dynamic-equation" data-equation="\[ M_a = ${M_a} \]" />

Note that the manifold has $E$ columns, and the number of rows depends on the number of observations in the $a$ time series.

!!! question "Why does this look backwards?"
    You may wonder why we have each point in the manifold going backwards in time when reading left-to-right.
    This is simply an unfortunate convention in the EDM literature which we adhere to.