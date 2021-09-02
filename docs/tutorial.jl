### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 7499784c-0ffd-48af-9b0d-d6937bcf9b0a
begin
	using DataFrames
	using LaTeXStrings
	using Latexify
	using SymEngine
	using PlutoUI
	using Random
	using Statistics
	using Printf
end

# ╔═╡ 99f564e2-611e-4138-8003-343077495a17
md"# EDM tutorial"

# ╔═╡ 3ffe80af-0484-46e6-a4d6-bcfd4f0ca2f7
md"""
This page has interactive sliders to adjust some hyperparameters.
Press the "Edit or run this notebook" button to launch this page inside a Binder instance to enable this functionality. This may take some time to launch, and it might be necessary to run (pressing Shift+Enter) the first cell manually.
"""

# ╔═╡ 6c40aa11-f636-4af3-bd11-53f4f21e4846
md"### Start with the data"

# ╔═╡ 2056d35f-b594-4ba6-b5ee-5f7db829194a
md"""
Imagine that we observe a time series $x$.
In tabular form, the data looks like:
"""

# ╔═╡ a9be623f-ada5-46c4-bdf3-d0a89a9812c9
begin
	obs = 12
	t = [symbols("t_$i") for i in 1:obs]
	x = [symbols("x_$i") for i in 1:obs]
	y = [symbols("y_$i") for i in 1:obs]
	z = [symbols("z_$i") for i in 1:obs]
	
	traw = [symbols(raw"t_{" * Base.string(i) * raw"}") for i in 1:obs]
	xraw = [symbols(raw"x_{" * Base.string(i) * raw"}") for i in 1:obs]
	#yraw = [symbols(raw"y_{" * Base.string(i) * raw"}") for i in 1:obs]
	#zraw = [symbols(raw"z_{" * Base.string(i) * raw"}") for i in 1:obs]
	
	#df = DataFrame(t = latexify.(traw), x = latexify.(xraw), y = latexify.(yraw), z = latexify.(zraw))
	df = DataFrame(t = latexify.(traw), x = latexify.(xraw))
end

# ╔═╡ 489ec795-acf4-4096-9b80-25a1c29efe77
md"So each one of $x_i$ is an *observation* of the $x$ time series."

# ╔═╡ 578cb599-4b44-4501-9798-e9d79e43b82c
md"### Creating a time-delayed embedding"

# ╔═╡ b9b9315b-32b2-45a8-86af-c96585c88f00
md"""
To create a time-delayed embedding based on any of these time series, we first need to choose the size of the embedding $E$.
"""

# ╔═╡ 569d0959-af43-4c88-a733-da272897b4bf
md"""
The data may be too finely sampled in time.
So we select a $\tau$ which means we only look at every $\tau$th observation for each time series.
"""

# ╔═╡ 92dfa63d-3799-491b-b2a6-512a44a98f09
md"Choose a value for $E$:"

# ╔═╡ 8630c42d-1357-4d7e-8d26-75aa5afe404a
begin 
	@bind E Slider(2:4; default=3, show_value=true)
end

# ╔═╡ afd006ee-2600-4a0b-a475-4172d22e4f6f
md"Choose a value for $\tau$:"

# ╔═╡ ce054612-97a2-48a0-9a49-4664d708823c
@bind τ Slider(1:3; default=2, show_value=true)

# ╔═╡ 6f9345ef-a557-4271-a9a7-23cdef85c98b
md"""
The time-delayed embedding of the $x$ time series with the given size E = $E and τ = $τ, is the manifold:
"""

# ╔═╡ 1b45974c-7a5f-47a7-996d-47d94dbe231c
begin
	function manifold(x, E, tau)
		Mrows = [reshape(x[(i + tau*(E-1)):-tau:(i)], 1, E) for i = 1:(obs-(E-1)*tau)]
		reduce(vcat, Mrows)
	end;
	
	function manifold_set(M)
		mset = raw"\{"
		
		maxPoints = 3
		numPoints = min(size(M, 1), maxPoints)
		
		for i = 1:numPoints
			if i > 1
				mset *= ", "
			end
			
			point = M[[i],:] 
			#point = reshape(point, 1, E)
			mset *= latexify(point, env=:raw)
			
			#pointTimes = (i + tau*(E-1)):-tau:(i)
			
			# mset *= raw"\{"
			# for j = 1:size(pointTimes, 1)
			# 	if j > 1
			# 		mset *= ", "
			# 	end
			# 	mset *= raw"x_{" * string(pointTimes[j]) * raw"}"
			# end
			# mset *= raw"\}"		
		end
		
		if size(M,1) > numPoints
			mset *= raw", \dots "
		end
		
		mset *= raw"\}"
			
		mset
	end;
	
	M_x = manifold(x, E, τ);
	
	M_x_set = manifold_set(M_x);
	
	L"M_x := \text{Manifold}(x, E,\tau) = %$M_x_set"
end

# ╔═╡ 279b6d77-00f8-40d4-b4cb-386b5d29dbdb
md"""
The manifold is a collection of these time-delayed *embedding vectors*.
For short, we just refer to each vector as a *point* on the manifold.
While the manifold notation above is the most accurate (a set of vectors) we will henceforward use the more convenient matrix notation:
"""

# ╔═╡ 152aa84f-71e7-446a-b435-b5daa57fd4b1
begin
	M_x_str = latexify(M_x)
	L"M_x := \text{Manifold}(x, E,\tau) = %$(M_x_str)"
end

# ╔═╡ acce14af-220b-478d-a8bc-0e06c5fa17b1
md"Note that the manifold has $E$ columns, and the number of rows depends on the number of observations in the $x$ time series."

# ╔═╡ 7e2459b4-70fe-489c-958e-b6476ffaac9f
md"### What does `edm explore x` do?"

# ╔═╡ 5cd973e8-078e-4d7e-9ddd-1a227e44801c
md"##### First split into library and prediction sets"

# ╔═╡ 9b69d2ce-7612-4588-9b43-44a1f99c0314
md"""
Firstly, the manifold $M_x$ is split into two separate parts, called the *library set* denoted $\mathscr{L}$ and the *prediction set* denoted $\mathscr{P}$.
By default, it takes the points of the $M_x$ and randomly assigns half of them to $\mathscr{L}$ and the other half to $\mathscr{P}$.
In this case we create a partition of the manifold, though if Stata is given other options then the same point may appear in both sets.
"""

# ╔═╡ b5df41e1-5d14-45be-abbf-0611d3c0d0cc
md"Starting with the time-delayed embedding of $x$:"

# ╔═╡ 82803a56-fd0b-49cc-a7b2-d86ddccb8865
latexalign(["M_x"], [M_x])

# ╔═╡ c76bfd49-486d-48c9-936e-e19d06669dfe
begin
	rng = MersenneTwister(1234);
	unifs = rand(rng, size(M_x, 1));
	med = median(unifs);
	
	libraryPoints = findall(unifs .<= med)
	predictionPoints = findall(unifs .> med)
	
	L = M_x[libraryPoints,:];
	P = M_x[predictionPoints,:];
	
	md"Then we may take the points $(Base.string(libraryPoints)) to create the library set, which leaves the remaining $(Base.string(predictionPoints)) points to create the prediction set."
end

# ╔═╡ 7c0c12f9-e348-4af3-a4d7-323bba09c95f
md"In that case, the *library set* is"

# ╔═╡ 45fa3658-3855-48b4-9eaa-a9a9bac55f9d
begin

	latexalign([LaTeXString(raw"\mathscr{L}")], [L])
end

# ╔═╡ a0d44f17-df31-4662-9777-f867fcfa44c7
md"and the *prediction set* is"

# ╔═╡ 8a934063-6348-48cb-9c30-532b44052fd9
latexalign([LaTeXString(raw"\mathscr{P}")], [P])

# ╔═╡ 6219fd69-62b7-4736-8fc2-82b7fd062da2
md"""
It will help to introduce a notation to refer to a specific point in these sets based on its row number.
E.g. in the example above, the first point in the library set takes the value:
"""

# ╔═╡ 963a24c4-743a-484e-a50d-8c5605c7f98f
latexalign([LaTeXString(raw"\mathscr{L}_1")], [L[[1],:]])

# ╔═╡ d7e2fa8d-7b9b-484d-91f2-a4237babc707
md"""
More generally $\mathscr{L}_{i}$ refers to the $i$th point in $\mathscr{L}$
and similarly $\mathscr{P}_{j}$ refers to the $j$th point in $\mathscr{P}$.
"""

# ╔═╡ 3de1fca8-13ac-4358-bb41-ce947bbce460
# begin
# t_all = manifold(t, E, τ)
# firstTime = t_all[trainingSlices[1],1]
	
# Markdown.parse(raw"The first observation in this slice was measured at time $" * Base.string(firstTime) * raw"$.")
# md"TODO: Alternative notation based on time."
# end;

# ╔═╡ 0fe181b0-c244-4b67-97a0-cf452aa6f277
md"##### Next, look at the future values of each point"

# ╔═╡ aaa5b419-65ca-41e0-a0fe-34af564c09d3
md"""
Each point on the manifold refers to a small trajectory of a time series, and for each point we look $p$ observations into the future of the time series.
"""

# ╔═╡ faf7fd88-aa9f-4b87-9e44-7a7dbfa14927
md"Choose a value for $p$:"

# ╔═╡ 2aa0c807-f3ab-426f-bf3b-6ec87c4ad2e9
@bind p Slider(-2:4; default=1, show_value=true)

# ╔═╡ 20628228-a140-4b56-bc63-5742660822c7
md"We have $p$ = $p."

# ╔═╡ 07bbe29b-41ab-4b9b-9c37-7a93f7c7f1fb
md"""
So if we take the first point of the prediction set $\mathscr{P}_{1}$ and say that $y_1^{\mathscr{P}}$ is the value it takes $p$ observations in the future, we get:
"""

# ╔═╡ 7b4b123c-6c91-48e4-9b08-70df7049f304
begin
	ahead = p
	x_fut = [symbols("x_$(i + τ*(E-1) + ahead)") for i = 1:(obs-(E-1)*τ)]
	y_fut = [symbols("y_$(i + τ*(E-1) + ahead)") for i = 1:(obs-(E-1)*τ)]
	z_fut = [symbols("z_$(i + τ*(E-1) + ahead)") for i = 1:(obs-(E-1)*τ)]
	
	x_fut_train = x_fut[libraryPoints]
	x_fut_pred = x_fut[predictionPoints]
	
	first_P_point = latexify(P[[1],:], env=:raw)
	first_y_P = latexify(x_fut_pred[[1],:], env=:raw)
	
	L"\mathscr{P}_{1} = %$first_P_point \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y_1^{\mathscr{P}} = %$first_y_P"
end

# ╔═╡ e9f56664-6d86-4a9c-b1a5-65d0af9bf1b0
md"""
This $p$ may be thought of as the *prediction horizon*, and in `explore` mode is defaults to τ and in `xmap` mode it defaults to 0.
"""

# ╔═╡ ad158c77-8a83-4ee0-9df6-bff0852ff896
md"""
In the literature, instead of measuring the number of observations $p$ ahead, authors normally use the value $T_p$ to denote the amount of time this corresponds to.
When data is regularly sampled (e.g. $t_i = i$) then there is no difference (e.g. $T_p = p$), however for irregularly sampled data the actual time difference may be different for each prediction.
"""

# ╔═╡ e4f62eab-a061-41a1-82a6-cf1cf4860ddf
md"In the training set, this means each point of $\mathscr{L}$ matches the corresponding value in $y^{\,\mathscr{L}}$:"

# ╔═╡ bf3906d4-729b-4f35-b0a1-b1eff5797602
begin
	L_str = latexify(L, env=:raw)
	y_L_str = latexify(x_fut_train, env=:raw)
	
 	L"\mathscr{L} = %$L_str \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y^{\,\mathscr{L}} = %$y_L_str"
end

# ╔═╡ 0c676a7d-1b7a-432b-8058-320a37188ab3
md"Similarly, for the prediction set:"

# ╔═╡ 7ac81a86-de83-4fb8-9415-5a8d71d58ca4
begin
	P_str = latexify(P, env=:raw)
	y_P_str = latexify(x_fut_pred, env=:raw) # [1:size(P,1)]
	L"\mathscr{P} = %$P_str \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y^{\mathscr{P}} = %$y_P_str"
end

# ╔═╡ b26cfd13-0749-4986-97d7-0dffc899757b
md"""
We may refer to elements of the $y^{\mathscr{L}}$ vector as *projections* as they come about by taking the $x$ time series and projecting it into the future by $p$ observations.
"""

# ╔═╡ 71fdd13f-19de-46e5-b11f-3e2824275505
md"### What does `edm explore x` predict?"

# ╔═╡ 7be9628d-4f92-4d4a-a1b1-a77141e77c30
md"""
When running `edm explore x`, we pretend that we don't know the values in $y^{\mathscr{P}}$ and that we want to predict them given we know $\mathscr{P}$, $\mathscr{L}$ and $y^{\,\mathscr{L}}$.
"""

# ╔═╡ ed5c583a-8cc2-4bdf-9696-fcc58dcb22fb
md"""
The first prediction is to try to find the value of $y_1^{\mathscr{P}}$ given the corresponding point $\mathscr{P}_1$:
"""

# ╔═╡ b0cb6b70-9c50-48d6-8e96-15b591d53221
L"\mathscr{P}_{1} = %$(latexify(P[[1],:], env=:raw)) \quad \underset{\small \text{Matches}}{\Rightarrow} \quad y_1^{\mathscr{P}} = \, ???"

# ╔═╡ 1e4c550e-ae5a-4fbd-867c-88e5b8013397
md"""
The terminology we use is that $y_1^{\mathscr{P}}$ is the *target* and the point $\mathscr{P}_1$ is the *predictee*.
"""

# ╔═╡ 216f4400-2b75-4472-9661-c477d8931d45
md"""
Looking over all the points in $\mathscr{L}$, we find the indices of the $k$ points which are the most similar to $\mathscr{P}_{1}$.

Let's pretend we have $k=2$ and the most similar points are $\mathscr{L}_{3}$ and $\mathscr{L}_{5}$.
We will choose the notation $\mathcal{NN}_k(1) = \{ 3, 5 \}$ to describe this set of $k$ nearest neighbours of $\mathscr{P}_{1}$.
"""

# ╔═╡ f29669bf-e5e1-4828-a4b0-311f5665a9c3
md"###### Using the simplex algorithm
Then, if we have chosen the `algorithm` to be the simplex algorithm, we predict that
"

# ╔═╡ 535aaf80-9130-4417-81ef-8031da2f7c73
L"y_{1}^{\mathscr{P}} \approx \hat{y}_1^{\mathscr{P}} := w_1 \times y_{3}^{\,\mathscr{L}} + w_2 \times y_{5}^{\,\mathscr{L}}"

# ╔═╡ 69dee0fe-4266-4444-a0f1-44db01b38dbd
md"""where $w_1$ and $w_2$ are some weights with add up to 1. Basically we are predicting that $y_1^{\mathscr{P}}$ is a weighted average of the points $\{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(1)}$."""

# ╔═╡ 657baf6f-4e4b-408c-918d-f007211699ea
md"To summarise the whole simplex procedure:"

# ╔═╡ d790cbae-85ed-46b7-b0c2-75568802f115
begin
	extractStr = raw"\underset{\small \text{Extracts}}{\Rightarrow} "
	getStr = raw"\underset{\small \text{Get predictee}}{\Rightarrow} "
	findInLStr = raw"\underset{\small \text{Find neighbours in } \mathscr{L}}{\Rightarrow} "
	predictStr = raw"\underset{\small \text{Make prediction}}{\Rightarrow} "
	
	L"
	\begin{align}
	\text{For target }y_i^{\mathscr{P}}
	&%$getStr
	\mathscr{P}_{i}
	%$findInLStr
	\mathcal{NN}_k(i) \\
	&\,\,\,\,%$extractStr
	\{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)}
	%$predictStr
	\hat{y}_i^{\mathscr{P}}
	\end{align}"
end

# ╔═╡ 28e51576-d9bb-46ff-bf24-4c16736b625c
md"###### Using the S-map algorithm
If however we chose `algorithm` to be the S-map algorithm, then we predict
"

# ╔═╡ 0b9081dd-5100-4232-a6be-c2d3d8e3f66f
L"y_{1}^{\mathscr{P}} \approx \hat{y}_1^{\mathscr{P}} := \sum_{j=1}^E w_{1j} \times  \mathscr{P}_{1j}"

# ╔═╡ 0a0df400-ca3f-4c5a-82b6-a536671e7d51
md"""
where the $\{ w_{1j} \}_{j=1,\cdots,E}$ weights are calculated by solving a linear system based on the points in $\{ \mathscr{L}_j \}_{j \in \mathcal{NN}_k(1)}$ and $\{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(1)}$.
"""

# ╔═╡ bd345675-ab21-4bdf-bbf3-99966a3d46bd
md"Given the specific"

# ╔═╡ fb293078-489d-4f86-afe2-abf75040af6d
L"\mathscr{P}_1 = %$(latexify(P[[1],:]))"

# ╔═╡ 2265c861-f5cf-468b-a523-e7352359c17f
md"in this example, the prediction would look like:"

# ╔═╡ 9c7a4ecb-37bb-448e-8d13-f608725a1f2e
begin
	weights = [symbols("w_{1$i}") for i in 1:E]
	weightedSum = sum(weights.*P[1,:])
	L"y_{1}^{\mathscr{P}} \approx \hat{y}_1^{\mathscr{P}} :=  %$weightedSum"
end

# ╔═╡ 94660506-0de1-4013-b7bf-79f49e09820b
md"To summarise the whole S-map procedure:"

# ╔═╡ 5d65f348-3a8e-4185-b9a1-24c5dec2303f
begin
	weightStr = raw"\underset{\small \text{Calculate}}{\Rightarrow}"
	
	L"
	\begin{align}
	\text{For target }y_i^{\mathscr{P}}
	&%$getStr
	\mathscr{P}_{i}
	%$findInLStr
	\mathcal{NN}_k(i) \\
	&\,\,\,\,%$extractStr
	\{ \mathscr{L}_j, y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)}
	%$weightStr
	\{ w_{ij} \}_{j=1,\ldots,E} 
	%$predictStr
	\hat{y}_i^{\mathscr{P}}
	\end{align}"
end

# ╔═╡ f6e12684-be4a-4fa9-8a2a-3ea6b205fe9a
md"##### Assessing the prediction quality"

# ╔═╡ ec5e9c90-f7e8-49fa-a7c5-36fc527ebb1d
md"""
We calculate the $\hat{y}_i^{\mathscr{P}}$ predictions for each target in the prediction set (so $i = 1, \dots, |\mathscr{P}|$), and store the predictions in a vector $\hat{y}^{\mathscr{P}}$.

As we have the true value of $y_i^{\mathscr{P}}$ for each target in the prediction set, we can compare our $\hat{y}_i^{\mathscr{P}}$ predictions and assess their quality using their correlation
"""

# ╔═╡ 6b7788e1-4dfa-4249-b3ae-9323c144d8a5
L" \rho := \mathrm{Correlation}(y^{\mathscr{P}} , \hat{y}^{\mathscr{P}}) "

# ╔═╡ e14e6991-ab3e-4ee3-84df-65474d682f95
md"""
or using the mean absolute error
"""

# ╔═╡ 98e1b0ae-da11-4abc-b82a-39f57d86eafb
L"\text{MAE} := \frac{1}{| \mathscr{P} |} \sum_{i=1}^{| \mathscr{P} |} | y_i^{\mathscr{P}} - \hat{y}_i^{\mathscr{P}} | ."

# ╔═╡ 9f05a08a-a2ff-464b-9399-04047823b568
md"### What does `edm xmap u v` do?"

# ╔═╡ 95058743-a0f0-4e40-998a-959fb3bcf98f
md"""
Imagine that we use the command:

`edm xmap u v, oneway`

This will consider two different time series, here labelled  $u$ and $v$.
The lagged embedding $M_u$ is constructed:
"""

# ╔═╡ 203af6eb-040f-41c3-abc1-a7a574681825
begin
	u = [symbols("u_$i") for i in 1:obs]
	v = [symbols("v_$i") for i in 1:obs]
	
	M_u = manifold(u, E, τ);
	M_v = manifold(v, E, τ);
	L"u = %$(latexify(u, env=:raw)) \quad \Rightarrow \quad M_u = %$(latexify(M_u))"
end


# ╔═╡ a8130051-db98-46e6-97bb-f724d7a290d9
md"The library set is a random subset of $L$ points of $M_u$.
The library size parameter $L$ is set by the Stata parameter `library`."

# ╔═╡ 7138b826-ed29-4f6b-8067-bff029271852
md"Choose a value for $L$:"

# ╔═╡ 93613b85-ae63-4785-b5aa-caa51dab6b73
@bind library Slider(3:size(M_u,1); default=3, show_value=true)

# ╔═╡ 99927d27-b407-4cbd-97cb-395b996b65fb
md"The value of $L$ is $library."

# ╔═╡ 8165de9a-6db9-49c6-9dae-ffc0970f3615
begin
	cutoff = sort(unifs)[library]
	libraryPointsXmap = findall(unifs .<= cutoff)
	
	L_xmap = M_u[libraryPointsXmap,:]
	L_xmap_str = latexify(L_xmap, env=:raw)
	
	L"\mathscr{L} = %$L_xmap_str"
end

# ╔═╡ b30914fc-3bdd-40af-b586-8f172bdbac70
md"""
On the other hand, the prediction set will include every point of the $u$ embedding so:
"""

# ╔═╡ fc9450de-6665-4301-a458-a8c4159e3269
begin
	P_xmap = M_u
	L"\mathscr{P} = M_u = %$(latexify(P_xmap))"
end

# ╔═╡ cc543a65-f38f-412b-a05e-3b7d5cabe573
md"""
The $v$ time series will be the values which we try to predict.
Here we are trying to predict $p$ observations ahead, where the default case is actually $p = 0$.
The $p = 0$ case means we are using the $u$ time series to try to predict the contemporaneous value of $v$.
A negative $p$ may be chosen, though this is a bit abnormal.
"""

# ╔═╡ 9cad1bf1-9ad9-4683-b912-84d779827cea
md"Choose a value for $p$:"

# ╔═╡ 70af044f-4375-4c2b-936b-0acc62a4db4e
@bind p_xmap Slider(-2:5; default=0, show_value=true)

# ╔═╡ 69816a32-c6c5-4e30-94f1-31727f567b73
md"The value of $p$ is $p_xmap."

# ╔═╡ 797b647c-f8e1-4681-b5b2-a8721aad940f
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

# ╔═╡ 25720cde-ce24-4ceb-be88-539837b5bf39
begin
	P_xmap
	matchStr
	y_P_str_xmap
L"\mathscr{P} = %$(latexify(P_xmap, env=:raw)) \quad %$matchStr \quad y^{\mathscr{P}} = %$y_P_str_xmap"
end

# ╔═╡ 1cff64b5-b3d9-41ce-aac0-c722821dda93
md"""
The prediction procedure is then the same as previous times, though the library and prediction sets all contain values from the $u$ time series whereas the $y$ projection vectors contain (usually contemporaneous) values from the $v$ time series.
"""

# ╔═╡ 15092345-e81a-4c28-81ae-e79e9d823853
begin
	kNNStr = raw"\underset{\small \text{Find neighbours in}}{\Rightarrow} "
	L"
	\underbrace{ \text{For target }y_i^{\mathscr{P}} }_{\text{Based on } v}
	%$getStr
	\underbrace{ \mathscr{P}_{i} }_{\text{Based on } u}
	%$kNNStr
	\underbrace{ \mathscr{L} }_{\text{Based on } u}
	%$matchStr
	\underbrace{ \{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)} }_{\text{Based on } v}
	%$predictStr
	\underbrace{ \hat{y}_i^{\mathscr{P}} }_{\text{Based on } v}"
end

# ╔═╡ 6fae5864-ec11-4e94-98a9-ffc0fd421049
md"## Adding more data to the manifold"

# ╔═╡ 6ea2e5d4-ff50-4efe-b80a-29940e7455f9
md"""
It can be advantageous to combine data from multiple sources into a single EDM 
analysis.

The `extra` command will incorporate additional pieces of data into the manifold.
As an example, the Stata command

`edm explore x, extra(y)`

will construct a manifold:
"""

# ╔═╡ 982d9067-da5b-4cbf-bcba-5c94ed2479b9
begin
	function manifold_with_extra(x, E, tau, extra)
		Mrows = [reshape(x[(i + tau*(E-1)):-tau:(i)], 1, E) for i = 1:(obs-(E-1)*tau)]
		M_x = reduce(vcat, Mrows)
		
		extraCol = [extra[(i + tau*(E-1))] for i = 1:(obs-(E-1)*tau)]
		
		hcat(M_x, extraCol)
	end;
	
	M_x_extra = manifold_with_extra(x, E, τ, y);
	
	#M_x_extra_set = manifold_set(M_x_extra);
	#L"M_{x,y} := %$M_x_extra_set"
	
	L"M_{x,y} := %$(latexify(M_x_extra))"
end

# ╔═╡ 3ddc8e19-b61a-4ad8-9d6c-1c38dbe7308d
md"""
After extra variables are added, the manifold $M_{x,y}$ no longer has $E$ columns.
In these cases, we make a distinction between $E$ which selects the number of lags for each time series, and the *actual* $E$ which is size of each point (i.e. the number of columns).
"""

# ╔═╡ 12dff36b-59da-45bc-984d-a5063f204bd9
md"""
By default just one $y$ observation is added to each point in the manifold. 

If $E$ lags of $y$ are required, then the command should be altered slightly to

`edm explore x, extra(y(e))`

and then the manifold will be:
"""

# ╔═╡ 66de25a9-4ed3-42e2-ae9f-427684a8fb1e
begin
	M_extras = manifold(y, E, τ);
	M_x_extras = hcat(M_x, M_extras)	
	#M_x_extras_set = manifold_set(M_x_extras)
	#L"M_{x,y} := %$M_x_extras_set"
	L"M_{x,y} := %$(latexify(M_x_extras))" 
end

# ╔═╡ dfdb3ef6-35b6-48f5-ae30-12b41d247c90
md"""
More than one `extra` variable can be added.
If some extras are lagged extra variables are specified after some unlagged extras, then the package will reorder them so that all the lagged extras are first.
"""

# ╔═╡ 1e2bd033-a948-4a60-9b78-13d4d88813f7
md"## What about irregularly sampled observations?"

# ╔═╡ 0c29af04-ebd1-4d10-bd10-819b7a3a94d9
md"""
We don't always have observations of real-world phenomena at regular intervals.
The `dt` option is designed to handle the situation when the times between each of the observations are irregular.

For example, when the command

`edm explore x, dt`

is specified, the manifold will be constructed as if the `extra` option had said to include the time differences.
"""

# ╔═╡ 365746a0-2f0c-481f-b8b3-00774d981196
md"Choose a value for $E$:"

# ╔═╡ e694efa3-d618-4430-ac5b-d1922f5b4c1e
@bind E_dt Slider(2:3; default=2, show_value=true)

# ╔═╡ 8dcfb9ac-1f74-4cf7-8419-13b5cddcef74
md"Choose a value for $\tau$:"

# ╔═╡ 710cf6bf-7eac-478a-9783-4eed31c2360e
@bind τ_dt Slider(1:4; default=2, show_value=true)

# ╔═╡ f9f0a2fa-613e-48d3-ac8a-bae77a60f22f
md"In this case ($E$ = $E_dt, τ = $τ_dt), the manifold will look like:"

# ╔═╡ b87518c1-1bae-4496-bae2-6d2bdaa8ad69
begin

	function manifold_set_dt(n, E, tau)
		mset = raw"\{"
		
		maxPoints = 2
		numPoints = min(n, maxPoints)
		
		for i = 1:numPoints
			if i > 1
				mset *= ", "
			end
			
			pointTimes = (i + tau*(E-1)):-tau:(i)
			pointTimes
			
			mset *= raw"["
			for j = 1:length(pointTimes)
				if j > 1
					mset *= raw"\,\,\,\,"
				end
				mset *= raw"x_{" * string(pointTimes[j]) * raw"}"
			end
			
			for j = 1:length(pointTimes)
				mset *= raw"\,\,\,\,"
				if pointTimes[j]+tau <= obs
					mset *= raw"(t_{" * string(pointTimes[j]+tau) * raw"}"
					mset *= raw"-t_{" * string(pointTimes[j]) * raw"})"
				else
					mset *= "?"
				end
			end
			
			mset *= raw"]  "		
		end
		
		if n > numPoints
			mset *= raw", \dots "
		end
		
		mset *= raw"\}"
			
		mset
	end;
	
	M_x_dt_set = manifold_set_dt(size(M_x, 1), E_dt, τ_dt)
	
	L"M_{x,\mathrm{d}t} := %$M_x_dt_set"
end

# ╔═╡ c1d4a958-9fc0-4eda-aaa6-f28fa88551d6
md"""If we define the notation that 

$dt_{i}^{\tau} = (t_{i+\tau} - t_i)$

then the manifold can be written as:
"""

# ╔═╡ 66a18830-0455-4c5a-9c96-719940d5e6c7
begin
	deltaT = [symbols(raw"dt_" * string(i)*raw"^τ") for i in 1:(obs)]
	M_x_dt = manifold(x, E_dt, τ_dt);
	M_dt = manifold(deltaT, E_dt, τ_dt);
	M_x_and_dt = hcat(M_x_dt, M_dt)	
	L"M_{x,\mathrm{d}t} := %$(latexify(M_x_and_dt))"
end

# ╔═╡ 9b15ced5-e4de-43e4-b1e5-9f380a788d34
md"""
If the *cumulative dt* option `cumdt` option is set, then the time differences are relative to the time of the prediction. That is
"""

# ╔═╡ 65d93d7f-f7f9-4598-9064-de8fbf490337
begin

	function manifold_set_cumdt(n, E, tau)
		mset = raw"\{"
		
		maxPoints = 2
		numPoints = min(n, maxPoints)
		
		for i = 1:numPoints
			if i > 1
				mset *= ", "
			end
			
			pointTimes = (i + tau*(E-1)):-tau:(i)
			pointTimes
			
			mset *= raw"["
			for j = 1:length(pointTimes)
				if j > 1
					mset *= raw"\,\,\,\,"
				end
				mset *= raw"x_{" * string(pointTimes[j]) * raw"}"
			end
			
			for j = 1:length(pointTimes)
				mset *= raw"\,\,\,\,"
				if pointTimes[j]+tau <= obs
					mset *= raw"(t_{" * string(pointTimes[1]+p) * raw"}"
					mset *= raw"-t_{" * string(pointTimes[j]) * raw"})"
				else
					mset *= "?"
				end
			end
			
			mset *= raw"]  "		
		end
		
		if n > numPoints
			mset *= raw", \dots "
		end
		
		mset *= raw"\}"
			
		mset
	end;
	
	M_x_cumdt_set = manifold_set_cumdt(size(M_x, 1), E_dt, τ_dt)
	
	L"M_{x,\Delta t} := %$M_x_cumdt_set"
end

# ╔═╡ 8c23501b-2bdd-4652-bb4c-8012772b758f
md"""If we define the notation that 

$\overline{\Delta} t_{ij}^{\tau p} = (t_{i+p} - t_{i-j\tau})$

then the manifold can be written as:
"""

# ╔═╡ 95998f9b-0c04-47b6-b75e-d90582f56db8
# begin
# 	cumdeltaT = [symbols(raw" ̄Δ̄t_" * string(i)*raw"^τ") for i in 1:(obs)]
# 	M_cumdt = manifold(cumdeltaT, E, τ);
# 	M_x_cumdt = hcat(M_x, M_cumdt)	
# 	L"M_{x,\mathrm{d}t} := %$(latexify(M_x_cumdt))"
# end
md"TODO..."

# ╔═╡ 6a96e421-b6c8-4f4f-9f13-4c34d5443d1b
md"### What does `copredict` do in explore mode?"

# ╔═╡ 885b10fe-5585-462b-9109-b293863d67be
md"""
Imagine that we use the command:

`edm explore x, copredictvar(z) copredict(out)`

This will first do a normal

`edm explore x`

operation, then it will perform a second set of *copredictions*.
This brings in a second time series $z$, and specifies that the predictions made in copredict mode should be stored in the Stata variable named `out`.
"""

# ╔═╡ 37585472-6078-4020-98f2-89ea31dbd4b9
md"""
In coprediction mode, the training set will include the entirety of the $M_x$ manifold and its projections:
"""

# ╔═╡ 80712325-3f89-409f-b0d3-d8eca06eae02
begin
	M_x_str
	matchStr
	y_L_copred_str = latexify(x_fut, env=:raw)
L"\mathscr{L} = M_x = %$M_x_str \quad %$matchStr \quad y^{\,\mathscr{L}} = %$y_L_copred_str"
end

# ╔═╡ b53ea6c9-5aaf-41fe-aaf1-4b65214f6acf
md"""
In copredict mode the most significant difference is that we change $\mathscr{P}$ to be the $M_z$ manifold for the $z$ time series and $y^{\mathscr{P}}$ to:
"""

# ╔═╡ e84366f8-611c-4021-bd86-6f46780e1487
begin
	M_z = manifold(z, E, τ)
	P_copred_z = M_z
	P_copred_z_str = latexify(P_copred_z, env=:raw)
	
	matchStr
	
	y_P_copred_str = latexify(z_fut, env=:raw)
	L"\mathscr{P} = M_z = %$P_copred_z_str \quad %$matchStr \quad y^{\mathscr{P}} = %$y_P_copred_str"
end

# ╔═╡ 3db5cd4f-6c4e-4c4f-be04-2be39f568959
md"""
The rest of the simplex procedure is the same as before:
"""

# ╔═╡ fec658e0-d77c-45de-b906-b8b6b4942bad
L"
\underbrace{ \text{For target }y_i^{\mathscr{P}} }_{ \text{Based on } z }
%$getStr
\underbrace{ \mathscr{P}_{i} }_{ \text{Based on } z }
%$kNNStr
\mathscr{L}
%$matchStr
\{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)}
%$predictStr
\hat{y}_i^{\mathscr{P}}"

# ╔═╡ d4166207-8fdf-484a-a6bb-835a966373a4
md"### What does `copredict` do in xmap mode?"

# ╔═╡ 4c572fa2-1177-4067-b129-5f4cf0bd302a
md"""
Imagine that we use the command:

`edm xmap u v, oneway copredictvar(w) copredict(out)`

Now we combine three different time series to create the predictions in the `out` Stata variable.
"""

# ╔═╡ 2da0ba39-c039-4b80-8f8b-863107ccaf96
md"""
In this case, the training set contains all the points in $M_u$:
"""

# ╔═╡ 770da5db-c46c-4a1d-a3f3-7df02321da05
begin

	M_u_str = (latexify(P_xmap))

 	y_L_str_xmap_copred = latexify(v_fut, env=:raw)

	L"\mathscr{L} = M_u = %$M_u_str \quad %$matchStr \quad y^{\,\mathscr{L}} = %$y_L_str_xmap_copred"
end

# ╔═╡ d51ef4c6-b5fd-4ea4-84d7-6ce137be89e7
md"""
The main change in coprediction is the prediction set and the targets are based on the new $w$ time series:
"""

# ╔═╡ dd793d3f-45f2-4eca-bc2e-3eaa67b59e41
begin
	w = [symbols("w_$i") for i in 1:obs]
	M_w = manifold(w, E, τ);
	P_xmap_copred = M_w
	
	matchStr
	
	co_ahead = p_xmap
	w_fut = [symbols("w_$(i + τ*(E-1) + co_ahead)") for i = 1:(obs-(E-1)*τ)]
	y_P_str_xmap_copred = latexify(w_fut, env=:raw)


	L"\mathscr{P} = M_w = %$(latexify(P_xmap_copred, env=:raw)) \quad %$matchStr \quad y^{\mathscr{P}} = %$y_P_str_xmap_copred"
end

# ╔═╡ 027667be-91db-4e4b-b4f2-07306ea7e4c1
md"""
Finally, the simplex prediction steps are the same, with:
"""

# ╔═╡ 1e00a8f0-94b0-430b-81b0-9846913878f5
L"
\underbrace{ \text{For target }y_i^{\mathscr{P}} }_{\text{Based on } w}
%$getStr
\underbrace{ \mathscr{P}_{i} }_{ \text{Based on } w }
%$kNNStr
\underbrace{ \mathscr{L} }_{\text{Based on } u}
%$matchStr
\underbrace{ \{ y_j^{\,\mathscr{L}} \}_{j \in \mathcal{NN}_k(i)} }_{\text{Based on } v}
%$predictStr
\underbrace{ \hat{y}_i^{\mathscr{P}} }_{\text{Based on } v}"

# ╔═╡ a29ab0d0-ea70-480f-90f6-5418534b27df
md"""
### Missing data 

To explain how the package handles missing data given different options, it is easiest to work by example.

Let's say we have the following time series and $(NAN) represents a missing value:
"""

# ╔═╡ 12bbff2f-ac5d-4904-ad0c-3fd3efdb1243
begin
	tMiss = [ 1.0, 2.5, 3.0, 4.5, 5.0, 6.0 ]
  	xMiss = [ 11, 12, NaN, 14, 15, 16 ]
	yMiss = [ 12, NaN, 14, 15, 16, NaN ]

	dfMiss = DataFrame(t = tMiss, x = xMiss)
end

# ╔═╡ 7b2ff2ef-904c-45ab-9ef6-000118fa2d8a
md"Let's also fix $E = 2$, $\tau = 1$ and $p = 1$ for these examples."

# ╔═╡ a8b14249-19f1-4f21-aa47-e11b3c4e9ce7
md"""
Here we have one obviously missing value for $x$ at time 3.
However, there are some hidden missing values also.

By default, the package will assume that your data was measured at a regular time interval and will insert missing values as necessary to create a regular grid.
For example, the above time series will be treated as if it were:
"""

# ╔═╡ 1f26c940-a41a-48c2-80b4-f90a3c3efa3b
begin
	tMissFill = [ 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0 ]
  	xMissFill = [ 11,  NaN, NaN, 12, NaN, NaN, NaN, 14, 15, NaN, 16 ]
	dfMissFill = DataFrame(t = tMissFill, x = xMissFill)
end

# ╔═╡ f7f37b62-ef27-4e31-97ea-e78e6f839028
md"""
The manifold of $x$ and it's projections $y$ will have missing values in them:
"""

# ╔═╡ 43c39e42-4f21-4c57-8ac2-7d5d18f1745b
begin	
	M_x_miss = [ 11 NaN;
				 NaN 11;
				 NaN NaN;
				 12 NaN;
				 NaN 12;
				 NaN NaN;
				 NaN NaN;
				 14 NaN;
				 15 14;
				 NaN 15;
				 16 NaN]
	
	y_miss = [ NaN; NaN; 12; NaN; NaN; NaN; 14; 15; NaN; 16; NaN]
	
	L"M_x = %$(latexify(M_x_miss, env=:raw)) %$matchStr y = %$(latexify(y_miss, env=:raw))"
end

# ╔═╡ 50ee9c7d-51e2-4f6c-a2de-c4b543ac83ef
md"""
We can see that the original missing value, combined with some slightly irregular sampling, created a reconstructed manifold that is mostly missing values!

By default, the points which contain missing values *will not be added to the library or prediction sets*.

For example, if we let the library and prediction sets be as big as possible then we will have:
"""

# ╔═╡ 042db7fe-2936-4663-8893-a43142b6ffc1
begin
	P_valid = vec(.! any(isnan.(M_x_miss), dims=2) )
	L_valid = P_valid .& (.! isnan.(y_miss))
	
	# Override my best wishes and just exclude missing values from prediction set.
	P_valid = L_valid
	
	L_miss = M_x_miss[L_valid,:]
	y_L_miss = y_miss[L_valid]
	
	L"\mathscr{L} = %$(latexify(L_miss, env=:raw)) %$matchStr y^{\mathscr{L}} = %$(latexify(y_L_miss, env=:raw))"
end

# ╔═╡ 25eb0830-0ce1-47dc-8856-eda9a91a3473
begin
	P_miss = M_x_miss[P_valid,:]
	y_P_miss = y_miss[P_valid]
	
	L"\mathscr{P} = %$(latexify(P_miss, env=:raw)) %$matchStr y^{\mathscr{P}} = %$(latexify(y_P_miss, env=:raw))"
end

# ╔═╡ 92049e5d-9a54-4b99-93d6-2971b77a99fc
md"""
Here we see that the resulting sets are totally empty!

This is because for a point to be in the library or prediction set (with default options) it must be fully observed and the corresponding $y$ projection must also be observed.
"""

# ╔═╡ 1134c18a-76c7-4b9c-bff6-f795954f5db4
md"#### The `allowmissing` flag"

# ╔═╡ cc25480d-aae7-4462-9c45-0c4ba6838a06
md"""
If we set the `allowmissing` option, then a point is included in the manifold even with some missing values.
The only caveat to this rule is that points which are totally missing will always be discarded.


The largest possible library and prediction sets with `allowmissing` in this example would be:
"""

# ╔═╡ 420dae80-0f3b-445a-afa0-6492b4de9579
begin
	P_am_valid = vec(any(.! isnan.(M_x_miss), dims=2) )
	
	L_am_valid = P_am_valid .& (.! isnan.(y_miss))
	
	# Sad face
	P_am_valid = L_am_valid
	
	L_am_miss = M_x_miss[L_am_valid,:]
	y_L_am_miss = y_miss[L_am_valid]
	
	L"\mathscr{L} = %$(latexify(L_am_miss, env=:raw)) %$matchStr y^{\mathscr{L}} = %$(latexify(y_L_am_miss, env=:raw))"
end

# ╔═╡ f20109a8-7cdf-46f0-bb8d-36f1e16f6310
begin
	P_am_miss = M_x_miss[P_am_valid,:]
	y_P_am_miss = y_miss[P_am_valid]
	
	L"\mathscr{P} = %$(latexify(P_am_miss, env=:raw)) %$matchStr y^{\mathscr{P}} = %$(latexify(y_P_am_miss, env=:raw))"
end

# ╔═╡ b3296c31-ace4-4f8a-9eb4-9de85e15dd3d
md"""
This discussion is implicitly assuming the `algorithm` is set to the simplex algorithm.
When the S-map algorithm is chosen, then we cannot let missing values into the library set $\mathscr{L}$.
This may change in a future implementation of the S-map algorithm.
"""

# ╔═╡ 393fa77e-f65d-4a3c-9e88-431e5030486f
md"""
#### The `dt` flag

When we add `dt`, we tell the package to remove missing observations and to also add the time between the observations into the manifold.

So, in this example, instead of the observed time series being:
"""

# ╔═╡ 23fa3b59-8152-4be9-ad70-b815c9517548
dfMiss

# ╔═╡ 750bbba1-8af9-4760-a3fc-3f7fbf3bd329
md"the `dt` basically acts as if the supplied data were:"

# ╔═╡ aa10e962-f312-4897-964c-e33e8df9bf6f
begin
	tMissDT = [ 1.0, 2.5, 4.5, 5.0, 6.0 ]
	dtMissDT = [1.5, 2.0, 0.5, 1.0, NaN ]
  	xMissDT = [ 11, 12, 14, 15, 16 ]
	yMissDT = [ 12, 14, 15, 16, NaN ]

	dfMissDT = DataFrame(t = tMissDT, x = xMissDT, dt = dtMissDT)
end

# ╔═╡ 5a3af8a1-788e-424a-b68c-96c80f817886
md"The resulting manifold and projections are:"

# ╔═╡ 1d37c724-6f6d-4a7f-a8ad-aaae28bc32d5
begin
	M_x_miss_dt = [ 11 NaN 1.5 NaN;
				 12 11 2.0 1.5;
				 14 12 0.5 2.0;
				 15 14 1.0 0.5;
				 16 15 NaN 1.0]
	y_miss_dt = [ 12; 14; 15; 16; NaN]
	
	L"M_x = %$(latexify(M_x_miss_dt, env=:raw)) %$matchStr y = %$(latexify(y_miss_dt, env=:raw))"
end

# ╔═╡ 568660d0-af25-44a6-be84-0ee58ead31d3
md"The largest possible library and prediction sets with `dt` in this example would be:"

# ╔═╡ aa289a3f-10be-414e-ac4c-d2e331786d10
begin
	P_dt_valid = vec(.! any(isnan.(M_x_miss_dt), dims=2) )
	L_dt_valid = P_dt_valid .& (.! isnan.(y_miss_dt))
	
	L_miss_dt = M_x_miss_dt[L_dt_valid,:]
	y_L_miss_dt = y_miss_dt[L_dt_valid]
	
	L"\mathscr{L} = %$(latexify(L_miss_dt, env=:raw)) %$matchStr y^{\mathscr{L}} = %$(latexify(y_L_miss_dt, env=:raw))"
end

# ╔═╡ 259abc33-ef51-47d7-8c12-e307eb800b37
begin
	P_miss_dt = M_x_miss_dt[P_dt_valid,:]
	y_P_miss_dt = y_miss_dt[P_dt_valid]
	
	L"\mathscr{P} = %$(latexify(P_miss_dt, env=:raw)) %$matchStr y^{\mathscr{P}} = %$(latexify(y_P_miss_dt, env=:raw))"
end

# ╔═╡ 480e2e49-041b-4ba8-b0db-fa24121a4f06
md"""
#### Both `allowmissing` and `dt` flags

If we set both flags, we tell the package to allow missing observations and to also add the time between the observations into the manifold.

So the time series
"""

# ╔═╡ 04a3ad75-f2a2-414a-b745-c680448780e3
dfMiss

# ╔═╡ a3d9085c-33b5-4361-bc52-fcde45f4a13a
md"will generate the manifold"

# ╔═╡ 6649d53e-c358-4efd-a95f-393addbe630a
begin
	M_x_miss_am_dt = [ 11 NaN 1.5 NaN;
				       12 11 0.5 1.5;
				       NaN 12 1.5 0.5;
				       14 NaN 0.5 1.5;
				       15 14 1.0 0.5;
				       16 15 NaN 1.0]
	y_miss_am_dt = [ 12; NaN; 14; 15; 16; NaN]
	
	L"M_x = %$(latexify(M_x_miss_am_dt, env=:raw)) %$matchStr y = %$(latexify(y_miss_am_dt, env=:raw))"
end

# ╔═╡ 9d9b8e70-c59d-43cf-a941-ba2f37c57e2b
md"and the largest possible library and prediction sets would be"

# ╔═╡ ad2431ae-d3fb-4ce1-bd32-edcee1e22322
begin
	P_am_dt_valid = vec(any(.!  isnan.(M_x_miss_am_dt), dims=2) )
	L_am_dt_valid = P_am_dt_valid .& (.! isnan.(y_miss_am_dt))

	# Sad face
	P_am_dt_valid = L_am_dt_valid
	
	L_miss_am_dt = M_x_miss_am_dt[L_am_dt_valid,:]
	y_L_miss_am_dt = y_miss_am_dt[L_am_dt_valid]
	
	L"\mathscr{L} = %$(latexify(L_miss_am_dt, env=:raw)) %$matchStr y^{\mathscr{L}} = %$(latexify(y_L_miss_am_dt, env=:raw))"
end

# ╔═╡ 42e39981-cc8b-4eea-9dea-8d7cabbd23da
begin
	P_miss_am_dt = M_x_miss_am_dt[P_am_dt_valid,:]
	y_P_miss_am_dt = y_miss_am_dt[P_am_dt_valid]
	
	L"\mathscr{P} = %$(latexify(P_miss_am_dt, env=:raw)) %$matchStr y^{\mathscr{P}} = %$(latexify(y_P_miss_am_dt, env=:raw))"
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[compat]
DataFrames = "~1.2.2"
LaTeXStrings = "~1.2.1"
Latexify = "~0.15.6"
PlutoUI = "~0.7.9"
SymEngine = "~0.8.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "dfbf58e0e470c2fd70020ae2c34e2f17b9fd1e4c"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.2.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MPC_jll]]
deps = ["GMP_jll", "Libdl", "MPFR_jll", "Pkg"]
git-tree-sha1 = "583d9bc863ad491571369212b9a8047219a0015d"
uuid = "2ce0c516-f11f-5db3-98ad-e0e1048fbd70"
version = "1.1.0+0"

[[MPFR_jll]]
deps = ["Artifacts", "GMP_jll", "Libdl"]
uuid = "3a97d323-0669-5f0c-9066-3539efd106a3"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "94bf17e83a0e4b20c8d77f6af8ffe8cc3b386c0a"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "508822dca004bf62e210609148511ad03ce8f1d8"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.0"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[SymEngine]]
deps = ["Compat", "Libdl", "LinearAlgebra", "RecipesBase", "SpecialFunctions", "SymEngine_jll"]
git-tree-sha1 = "a27a507f5092a77b0619721c6f852a5f634fc52f"
uuid = "123dc426-2d89-5057-bbad-38513e3affd8"
version = "0.8.3"

[[SymEngine_jll]]
deps = ["GMP_jll", "Libdl", "MPC_jll", "MPFR_jll", "Pkg"]
git-tree-sha1 = "4dacada8e05ac49eb768219f8d02bc6b608627fb"
uuid = "3428059b-622b-5399-b16f-d347a77089a4"
version = "0.6.0+1"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─7499784c-0ffd-48af-9b0d-d6937bcf9b0a
# ╟─99f564e2-611e-4138-8003-343077495a17
# ╟─3ffe80af-0484-46e6-a4d6-bcfd4f0ca2f7
# ╟─6c40aa11-f636-4af3-bd11-53f4f21e4846
# ╟─2056d35f-b594-4ba6-b5ee-5f7db829194a
# ╟─a9be623f-ada5-46c4-bdf3-d0a89a9812c9
# ╟─489ec795-acf4-4096-9b80-25a1c29efe77
# ╟─578cb599-4b44-4501-9798-e9d79e43b82c
# ╟─b9b9315b-32b2-45a8-86af-c96585c88f00
# ╟─569d0959-af43-4c88-a733-da272897b4bf
# ╟─92dfa63d-3799-491b-b2a6-512a44a98f09
# ╟─8630c42d-1357-4d7e-8d26-75aa5afe404a
# ╟─afd006ee-2600-4a0b-a475-4172d22e4f6f
# ╟─ce054612-97a2-48a0-9a49-4664d708823c
# ╟─6f9345ef-a557-4271-a9a7-23cdef85c98b
# ╟─1b45974c-7a5f-47a7-996d-47d94dbe231c
# ╟─279b6d77-00f8-40d4-b4cb-386b5d29dbdb
# ╟─152aa84f-71e7-446a-b435-b5daa57fd4b1
# ╟─acce14af-220b-478d-a8bc-0e06c5fa17b1
# ╟─7e2459b4-70fe-489c-958e-b6476ffaac9f
# ╟─5cd973e8-078e-4d7e-9ddd-1a227e44801c
# ╟─9b69d2ce-7612-4588-9b43-44a1f99c0314
# ╟─b5df41e1-5d14-45be-abbf-0611d3c0d0cc
# ╟─82803a56-fd0b-49cc-a7b2-d86ddccb8865
# ╟─c76bfd49-486d-48c9-936e-e19d06669dfe
# ╟─7c0c12f9-e348-4af3-a4d7-323bba09c95f
# ╟─45fa3658-3855-48b4-9eaa-a9a9bac55f9d
# ╟─a0d44f17-df31-4662-9777-f867fcfa44c7
# ╟─8a934063-6348-48cb-9c30-532b44052fd9
# ╟─6219fd69-62b7-4736-8fc2-82b7fd062da2
# ╟─963a24c4-743a-484e-a50d-8c5605c7f98f
# ╟─d7e2fa8d-7b9b-484d-91f2-a4237babc707
# ╟─3de1fca8-13ac-4358-bb41-ce947bbce460
# ╟─0fe181b0-c244-4b67-97a0-cf452aa6f277
# ╟─aaa5b419-65ca-41e0-a0fe-34af564c09d3
# ╟─faf7fd88-aa9f-4b87-9e44-7a7dbfa14927
# ╟─2aa0c807-f3ab-426f-bf3b-6ec87c4ad2e9
# ╟─20628228-a140-4b56-bc63-5742660822c7
# ╟─07bbe29b-41ab-4b9b-9c37-7a93f7c7f1fb
# ╟─7b4b123c-6c91-48e4-9b08-70df7049f304
# ╟─e9f56664-6d86-4a9c-b1a5-65d0af9bf1b0
# ╟─ad158c77-8a83-4ee0-9df6-bff0852ff896
# ╟─e4f62eab-a061-41a1-82a6-cf1cf4860ddf
# ╟─bf3906d4-729b-4f35-b0a1-b1eff5797602
# ╟─0c676a7d-1b7a-432b-8058-320a37188ab3
# ╟─7ac81a86-de83-4fb8-9415-5a8d71d58ca4
# ╟─b26cfd13-0749-4986-97d7-0dffc899757b
# ╟─71fdd13f-19de-46e5-b11f-3e2824275505
# ╟─7be9628d-4f92-4d4a-a1b1-a77141e77c30
# ╟─ed5c583a-8cc2-4bdf-9696-fcc58dcb22fb
# ╟─b0cb6b70-9c50-48d6-8e96-15b591d53221
# ╟─1e4c550e-ae5a-4fbd-867c-88e5b8013397
# ╟─216f4400-2b75-4472-9661-c477d8931d45
# ╟─f29669bf-e5e1-4828-a4b0-311f5665a9c3
# ╟─535aaf80-9130-4417-81ef-8031da2f7c73
# ╟─69dee0fe-4266-4444-a0f1-44db01b38dbd
# ╟─657baf6f-4e4b-408c-918d-f007211699ea
# ╟─d790cbae-85ed-46b7-b0c2-75568802f115
# ╟─28e51576-d9bb-46ff-bf24-4c16736b625c
# ╟─0b9081dd-5100-4232-a6be-c2d3d8e3f66f
# ╟─0a0df400-ca3f-4c5a-82b6-a536671e7d51
# ╟─bd345675-ab21-4bdf-bbf3-99966a3d46bd
# ╟─fb293078-489d-4f86-afe2-abf75040af6d
# ╟─2265c861-f5cf-468b-a523-e7352359c17f
# ╟─9c7a4ecb-37bb-448e-8d13-f608725a1f2e
# ╟─94660506-0de1-4013-b7bf-79f49e09820b
# ╟─5d65f348-3a8e-4185-b9a1-24c5dec2303f
# ╟─f6e12684-be4a-4fa9-8a2a-3ea6b205fe9a
# ╟─ec5e9c90-f7e8-49fa-a7c5-36fc527ebb1d
# ╟─6b7788e1-4dfa-4249-b3ae-9323c144d8a5
# ╟─e14e6991-ab3e-4ee3-84df-65474d682f95
# ╟─98e1b0ae-da11-4abc-b82a-39f57d86eafb
# ╟─9f05a08a-a2ff-464b-9399-04047823b568
# ╟─95058743-a0f0-4e40-998a-959fb3bcf98f
# ╟─203af6eb-040f-41c3-abc1-a7a574681825
# ╟─a8130051-db98-46e6-97bb-f724d7a290d9
# ╟─7138b826-ed29-4f6b-8067-bff029271852
# ╟─93613b85-ae63-4785-b5aa-caa51dab6b73
# ╟─99927d27-b407-4cbd-97cb-395b996b65fb
# ╟─8165de9a-6db9-49c6-9dae-ffc0970f3615
# ╟─b30914fc-3bdd-40af-b586-8f172bdbac70
# ╟─fc9450de-6665-4301-a458-a8c4159e3269
# ╟─cc543a65-f38f-412b-a05e-3b7d5cabe573
# ╟─9cad1bf1-9ad9-4683-b912-84d779827cea
# ╟─70af044f-4375-4c2b-936b-0acc62a4db4e
# ╟─69816a32-c6c5-4e30-94f1-31727f567b73
# ╟─797b647c-f8e1-4681-b5b2-a8721aad940f
# ╟─25720cde-ce24-4ceb-be88-539837b5bf39
# ╟─1cff64b5-b3d9-41ce-aac0-c722821dda93
# ╟─15092345-e81a-4c28-81ae-e79e9d823853
# ╟─6fae5864-ec11-4e94-98a9-ffc0fd421049
# ╟─6ea2e5d4-ff50-4efe-b80a-29940e7455f9
# ╟─982d9067-da5b-4cbf-bcba-5c94ed2479b9
# ╟─3ddc8e19-b61a-4ad8-9d6c-1c38dbe7308d
# ╟─12dff36b-59da-45bc-984d-a5063f204bd9
# ╟─66de25a9-4ed3-42e2-ae9f-427684a8fb1e
# ╟─dfdb3ef6-35b6-48f5-ae30-12b41d247c90
# ╟─1e2bd033-a948-4a60-9b78-13d4d88813f7
# ╟─0c29af04-ebd1-4d10-bd10-819b7a3a94d9
# ╟─365746a0-2f0c-481f-b8b3-00774d981196
# ╟─e694efa3-d618-4430-ac5b-d1922f5b4c1e
# ╟─8dcfb9ac-1f74-4cf7-8419-13b5cddcef74
# ╟─710cf6bf-7eac-478a-9783-4eed31c2360e
# ╟─f9f0a2fa-613e-48d3-ac8a-bae77a60f22f
# ╟─b87518c1-1bae-4496-bae2-6d2bdaa8ad69
# ╟─c1d4a958-9fc0-4eda-aaa6-f28fa88551d6
# ╟─66a18830-0455-4c5a-9c96-719940d5e6c7
# ╟─9b15ced5-e4de-43e4-b1e5-9f380a788d34
# ╟─65d93d7f-f7f9-4598-9064-de8fbf490337
# ╟─8c23501b-2bdd-4652-bb4c-8012772b758f
# ╟─95998f9b-0c04-47b6-b75e-d90582f56db8
# ╟─6a96e421-b6c8-4f4f-9f13-4c34d5443d1b
# ╟─885b10fe-5585-462b-9109-b293863d67be
# ╟─37585472-6078-4020-98f2-89ea31dbd4b9
# ╟─80712325-3f89-409f-b0d3-d8eca06eae02
# ╟─b53ea6c9-5aaf-41fe-aaf1-4b65214f6acf
# ╟─e84366f8-611c-4021-bd86-6f46780e1487
# ╟─3db5cd4f-6c4e-4c4f-be04-2be39f568959
# ╟─fec658e0-d77c-45de-b906-b8b6b4942bad
# ╟─d4166207-8fdf-484a-a6bb-835a966373a4
# ╟─4c572fa2-1177-4067-b129-5f4cf0bd302a
# ╟─2da0ba39-c039-4b80-8f8b-863107ccaf96
# ╟─770da5db-c46c-4a1d-a3f3-7df02321da05
# ╟─d51ef4c6-b5fd-4ea4-84d7-6ce137be89e7
# ╟─dd793d3f-45f2-4eca-bc2e-3eaa67b59e41
# ╟─027667be-91db-4e4b-b4f2-07306ea7e4c1
# ╟─1e00a8f0-94b0-430b-81b0-9846913878f5
# ╟─a29ab0d0-ea70-480f-90f6-5418534b27df
# ╟─12bbff2f-ac5d-4904-ad0c-3fd3efdb1243
# ╟─7b2ff2ef-904c-45ab-9ef6-000118fa2d8a
# ╟─a8b14249-19f1-4f21-aa47-e11b3c4e9ce7
# ╟─1f26c940-a41a-48c2-80b4-f90a3c3efa3b
# ╟─f7f37b62-ef27-4e31-97ea-e78e6f839028
# ╟─43c39e42-4f21-4c57-8ac2-7d5d18f1745b
# ╟─50ee9c7d-51e2-4f6c-a2de-c4b543ac83ef
# ╟─042db7fe-2936-4663-8893-a43142b6ffc1
# ╟─25eb0830-0ce1-47dc-8856-eda9a91a3473
# ╟─92049e5d-9a54-4b99-93d6-2971b77a99fc
# ╟─1134c18a-76c7-4b9c-bff6-f795954f5db4
# ╟─cc25480d-aae7-4462-9c45-0c4ba6838a06
# ╟─420dae80-0f3b-445a-afa0-6492b4de9579
# ╟─f20109a8-7cdf-46f0-bb8d-36f1e16f6310
# ╟─b3296c31-ace4-4f8a-9eb4-9de85e15dd3d
# ╟─393fa77e-f65d-4a3c-9e88-431e5030486f
# ╟─23fa3b59-8152-4be9-ad70-b815c9517548
# ╟─750bbba1-8af9-4760-a3fc-3f7fbf3bd329
# ╟─aa10e962-f312-4897-964c-e33e8df9bf6f
# ╟─5a3af8a1-788e-424a-b68c-96c80f817886
# ╟─1d37c724-6f6d-4a7f-a8ad-aaae28bc32d5
# ╟─568660d0-af25-44a6-be84-0ee58ead31d3
# ╟─aa289a3f-10be-414e-ac4c-d2e331786d10
# ╟─259abc33-ef51-47d7-8c12-e307eb800b37
# ╟─480e2e49-041b-4ba8-b0db-fa24121a4f06
# ╟─04a3ad75-f2a2-414a-b745-c680448780e3
# ╟─a3d9085c-33b5-4361-bc52-fcde45f4a13a
# ╟─6649d53e-c358-4efd-a95f-393addbe630a
# ╟─9d9b8e70-c59d-43cf-a941-ba2f37c57e2b
# ╟─ad2431ae-d3fb-4ce1-bd32-edcee1e22322
# ╟─42e39981-cc8b-4eea-9dea-8d7cabbd23da
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
