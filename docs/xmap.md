# What does `edm xmap u v` do?

Choose a value for $E$:

<div class="slidecontainer"><input type="range" min="1" max="10" value="2" class="slider" id="E"></div>
The value of $E$ is <span class="E_choice"></span>

Choose a value for $\tau$:

<div class="slidecontainer"><input type="range" min="1" max="5" value="1" class="slider" id="tau"></div>
The value of $\tau$ is <span class="tau_choice"></span>

<!-- 
    begin
        M_x = manifold(x, E, τ);
        rng = MersenneTwister(1234);
        unifs = rand(rng, size(M_x, 1));
    end;
-->

Imagine that we use the command:

```stata
edm xmap u v, oneway
```

This will consider two different time series, here labelled  $u$ and $v$.
The lagged embedding $M_u$ is constructed:

<!-- 
    begin
        u = [symbols("u_$i") for i in 1:obs]
        v = [symbols("v_$i") for i in 1:obs]
        
        M_u = manifold(u, E, τ);
        M_v = manifold(v, E, τ);
        L"u = %$(latexify(u, env=:raw)) \quad \Rightarrow \quad M_u = %$(latexify(M_u))"
    end
-->

The library set is a random subset of $L$ points of $M_u$.
The library size parameter $L$ is set by the Stata parameter `library`.

Choose a value for $L$:


<div class="slidecontainer"><input type="range" min="3" max="10" value="3" class="slider" id="library"></div>
The value of $L$ is <span class="library_choice"></span>
<!-- Technically, max of this slider should be size(M_u, 1) -->

<!-- 
    begin
        cutoff = sort(unifs)[library]
        libraryPointsXmap = findall(unifs .<= cutoff)
        
        L_xmap = M_u[libraryPointsXmap,:]
        L_xmap_str = latexify(L_xmap, env=:raw)
        
        L"\mathscr{L} = %$L_xmap_str"
    end
-->

On the other hand, the prediction set will include every point of the $u$ embedding so:
<!-- 
    begin
        P_xmap = M_u
        L"\mathscr{P} = M_u = %$(latexify(P_xmap))"
    end
-->

The $v$ time series will be the values which we try to predict.
Here we are trying to predict $p$ observations ahead, where the default case is actually $p = 0$.
The $p = 0$ case means we are using the $u$ time series to try to predict the contemporaneous value of $v$.
A negative $p$ may be chosen, though this is a bit abnormal.

Choose a value for $p$:

<div class="slidecontainer"><input type="range" min="-5" max="5" value="1" class="slider" id="p"></div>
The value of $p$ is <span class="E_choice"></span>
<!-- 
    begin
        ahead_xmap = p_xmap
        v_fut = [symbols("v_$(i + τ*(E-1) + ahead_xmap)") for i = 1:(obs-(E-1)*τ)]
        v_fut_train = v_fut[libraryPointsXmap]
        v_fut_pred = v_fut
        y_L_str_xmap = latexify(v_fut_train[1:size(L_xmap,1)], env=:raw)
        y_P_str_xmap = latexify(v_fut_pred[1:size(P_xmap,1)], env=:raw)
        matchStr = raw"\underset{\small \text{Matches}}{\Rightarrow} "
        L"\mathscr{L} = %$L_xmap_str \quad %$matchStr \quad y^{\,\mathscr{L}} = %$y_L_str_xmap"
    end
-->

<!--
    begin
        P_xmap
        matchStr
        y_P_str_xmap
    L"\mathscr{P} = %$(latexify(P_xmap, env=:raw)) \quad %$matchStr \quad y^{\mathscr{P}} = %$y_P_str_xmap"
    end
 -->
 
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