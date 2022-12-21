# Missing data

To explain how the package handles missing data given different options, it is easiest to work by example.

Let's say we have the following time series and NA represents a missing value:

<center>

| $t$ | $a_t$ |
| :-: | :---: |
| 1.0 |  11   |
| 2.5 |  12   |
| 3.0 |  NA   |
| 4.5 |  14   |
| 5.0 |  15   |
| 6.0 |  16   |

</center>

Let's also fix $E = 2$, $\tau = 1$ and $p = 1$ for these examples.

Here we have one obviously missing value for $a$ at time 3.
However, there are some hidden missing values also.

By default, the package will assume that your data was measured at a regular time interval and will insert missing values as necessary to create a regular grid.

For example, the above time series will be treated as if it were sampled every half time unit.
So, when creating the $E=2$ manifold it will
<!-- 
<center>

| $t$ | $t - \frac12$ | $t + \frac12$ | $a_t$ | $a_{t-\frac12}$ | $a_{t+\frac12}$ |
| :-: | :-----------: | :-----------: | :---: | :-------------: | :-------------: |
| 1.0 |      0.5      |      1.5      |  11   |       NA        |       NA        |
| 2.5 |      2.0      |      3.0      |  12   |       NA        |       NA        |
| 3.0 |      2.5      |      3.5      |  NA   |       12        |       NA        |
| 4.5 |      4.0      |      5.0      |  14   |       NA        |       15        |
| 5.0 |      4.5      |      5.5      |  15   |       14        |       NA        |
| 6.0 |      5.5      |      6.5      |  16   |       NA        |       NA        |

</center> -->

<center>

| $t$ | $a_t$ |
| :-: | :---: |
| 1.0 |  11   |
| 1.5 |  NA   |
| 2.0 |  NA   |
| 2.5 |  12   |
| 3.0 |  NA   |
| 3.5 |  NA   |
| 4.0 |  NA   |
| 4.5 |  14   |
| 5.0 |  15   |
| 5.5 |  NA   |
| 6.0 |  16   |

</center>

The manifold of $a$ and it's projections $b$ will have missing values in them:

\[
  M_a = \left[\begin{array}{cc}
    11 & \text{NA} \\
    %\text{NA} & 11 \\
    %\text{NA} & \text{NA} \\
    12 & \text{NA} \\
    \text{NA} & 12 \\
    %\text{NA} & \text{NA} \\
    %\text{NA} & \text{NA} \\
    14 & \text{NA} \\
    15 & 14 \\
    %\text{NA} & 15 \\
    16 & \text{NA} \\
  \end{array}\right]
  \underset{\small \text{Matches}}{\Rightarrow}
  y = \left[ \begin{array}{c}
    \text{NA} \\
    %\text{NA} \\
    %12 \\
    \text{NA} \\
    \text{NA} \\
    %\text{NA} \\
    %14 \\
    15 \\
    \text{NA} \\
    %16 \\
    \text{NA} \\
  \end{array} \right]
\]

We can see that the original missing value, combined with some slightly irregular sampling, created a reconstructed manifold that is mostly missing values!

By default, the points which contain missing values _will not be added to the library or prediction sets_.

For example, if we let the library and prediction sets be as big as possible then we will have:

\[
  \mathscr{L} = \emptyset
  \underset{\small \text{Matches}}{\Rightarrow}
  y^{\mathscr{L}} = \emptyset
\]

\[
  \mathscr{P} = \left[ \begin{array}{cc}
    15 & 14 \\
  \end{array} \right]
  \underset{\small \text{Matches}}{\Rightarrow}
  y^{\mathscr{P}} = \left[ \begin{array}{c}
    \text{NA} \\
  \end{array} \right]
\]

Here we see that the library set is totally empty!
This is because for a point to be in the library (with default options) it must be fully observed and the corresponding $b$ projection must also be observed.
Similarly, the prediction set is almost empty because (with default options) it must be fully observed.

## The `allowmissing` flag

If we set the `allowmissing` option, then a point is included in the manifold even with some missing values.
The only caveats to this rule are:

- points which are totally missing will always be discarded,
- we can't have missing targets for points in the library set.

The largest possible library and prediction sets with `allowmissing` in this example would be:

\[
  \mathscr{L} = \left[\begin{array}{cc}
    14 & \text{NA} \\
  \end{array}\right]
  \underset{\small \text{Matches}}{\Rightarrow}
  y^{\mathscr{L}} = \left[ \begin{array}{c}
    15 \\
  \end{array} \right]
\]

\[
  \mathscr{P} = M_a = \left[\begin{array}{cc}
    11 & \text{NA} \\
    %\text{NA} & 11 \\
    %\text{NA} & \text{NA} \\
    12 & \text{NA} \\
    \text{NA} & 12 \\
    %\text{NA} & \text{NA} \\
    %\text{NA} & \text{NA} \\
    14 & \text{NA} \\
    15 & 14 \\
    %\text{NA} & 15 \\
    16 & \text{NA} \\
  \end{array}\right]
  \underset{\small \text{Matches}}{\Rightarrow}
  y^{\mathscr{P}} = y = \left[ \begin{array}{c}
    \text{NA} \\
    %\text{NA} \\
    %12 \\
    \text{NA} \\
    \text{NA} \\
    %\text{NA} \\
    %14 \\
    15 \\
    \text{NA} \\
    %16 \\
    \text{NA} \\
  \end{array} \right]
\]

This discussion is implicitly assuming the `algorithm` is set to the simplex algorithm.
When the S-map algorithm is chosen, then we cannot let missing values into the library set $\mathscr{L}$.
This may change in a future implementation of the S-map algorithm.

## The `dt` flag

When we add `dt`, we tell the package to remove missing observations and to also add the time between the observations into the manifold.

So, in this example, instead of the observed time series being:

<center>

| $t$ | $a_t$ |
| :-: | :---: |
| 1.0 |  11   |
| 2.5 |  12   |
| 3.0 |  NA   |
| 4.5 |  14   |
| 5.0 |  15   |
| 6.0 |  16   |

</center>

the `dt` basically acts as if the supplied data were:

<center>

| $t$ | $a_t$ | $\mathrm{d}t$ |
| :-: | :---: | :-----------: |
| 1.0 |  11   |      1.5      |
| 2.5 |  12   |      2.0      |
| 4.5 |  14   |      0.5      |
| 5.0 |  15   |      1.0      |
| 6.0 |  16   |      NA       |

</center>

The resulting manifold and projections are:

\[
   M_a = \left[\begin{array}{cccc}
    12 & 11 & 2.0 & 1.5 \\
    14 & 12 & 0.5 & 2.0 \\
    15 & 14 & 1.0 & 0.5 \\
  \end{array}\right]
  \underset{\small \text{Matches}}{\Rightarrow}
  y = \left[ \begin{array}{c}
    14 \\
    15 \\
    16 \\
  \end{array} \right]
\]

The largest possible library and prediction sets with `dt` in this example would be:

\[
   \mathscr{L} = \mathscr{P} = M_a = \left[\begin{array}{cccc}
    12 & 11 & 2.0 & 1.5 \\
    14 & 12 & 0.5 & 2.0 \\
    15 & 14 & 1.0 & 0.5 \\
  \end{array}\right]
  \underset{\small \text{Matches}}{\Rightarrow}
  y^{\mathscr{L}} = y^{\mathscr{P}} = y = \left[ \begin{array}{c}
    14 \\
    15 \\
    16 \\
  \end{array} \right]
\]

## Both `allowmissing` and `dt` flags

If we set both flags, we tell the package to allow missing observations and to also add the time between the observations into the manifold.

So our original time series

<center>

| $t$ | $a_t$ |
| :-: | :---: |
| 1.0 |  11   |
| 2.5 |  12   |
| 3.0 |  NA   |
| 4.5 |  14   |
| 5.0 |  15   |
| 6.0 |  16   |

</center>

will generate the manifold

\[
  M_a = \left[\begin{array}{cccc}
    11 & \text{NA} & 1.5 & \text{NA} \\
    12 & 11 & 0.5 & 1.5 \\
    \text{NA} & 12 & 1.5 & 0.5 \\
    14 & \text{NA} & 0.5 & 1.5 \\
    15 & 14 & 1.0 & 0.5 \\
    16 & 15 & \text{NA} & 1.0 \\
  \end{array}\right]
  \underset{\small \text{Matches}}{\Rightarrow}
  y = \left[ \begin{array}{c}
    12 \\
    \text{NA} \\
    14 \\
    15 \\
    16 \\
    \text{NA}
  \end{array} \right]
\]

and the largest possible library and prediction sets would be

\[
  \mathscr{L} = \left[\begin{array}{cccc}
    11 & \text{NA} & 1.5 & \text{NA} \\
    \text{NA} & 12 & 1.5 & 0.5 \\
    14 & \text{NA} & 0.5 & 1.5 \\
    15 & 14 & 1.0 & 0.5 \\
  \end{array}\right]
  \underset{\small \text{Matches}}{\Rightarrow}
  y^{\mathscr{L}} = \left[ \begin{array}{c}
    12 \\
    14 \\
    15 \\
    16 \\
  \end{array} \right]
\]

\[
  \mathscr{P} = M_a = \left[\begin{array}{cccc}
    11 & \text{NA} & 1.5 & \text{NA} \\
    12 & 11 & 0.5 & 1.5 \\
    \text{NA} & 12 & 1.5 & 0.5 \\
    14 & \text{NA} & 0.5 & 1.5 \\
    15 & 14 & 1.0 & 0.5 \\
    16 & 15 & \text{NA} & 1.0 \\
  \end{array}\right]
  \underset{\small \text{Matches}}{\Rightarrow}
  y^{\mathscr{P}} = y = \left[ \begin{array}{c}
    12 \\
    \text{NA} \\
    14 \\
    15 \\
    16 \\
    \text{NA}
  \end{array} \right]
\]
