const update_centered_equations = function () {
  // Read in the manifold specifications from the sliders
  const numObs = parseInt(document.getElementById("numObs").value);
  const E = parseInt(document.getElementById("E").value);
  const tau = parseInt(document.getElementById("tau").value);
  const p = 1;
  const allowMissing = false;

  // Construct the manifold and targets
  const M = manifold("x", numObs, E, tau, allowMissing, p);

  // Turn these into latex arrays
  const maniTex = latexify(M.manifold);

  // Split the manifolds into library and prediction sets
  const library = Math.floor(M.manifold.length / 2);

  const libSet = M.manifold.slice(0, library);
  const libTargets = M.targets.slice(0, library);

  const predSet = M.manifold.slice(library);
  const predTargets = M.targets.slice(library);

  // Convert to latex to hand to mathjax
  const libSetTex = latexify(libSet);
  const predSetTex = latexify(predSet);

  // Save the result to the page
  let eqnsToTypeset = [];

  const equations = document.querySelectorAll(".dynamic-equation");
  equations.forEach((eqn) => {
    const prevRenderedEquation = eqn.dataset.renderedEquation;
    const renderedEquation = eqn.dataset.equation
      .replace(/\${M_x}/, maniTex)
      .replace(/\${L}/, libSetTex)
      .replace(/\${P}/, predSetTex);

    if (prevRenderedEquation != renderedEquation) {
      eqnsToTypeset.push(eqn);
      MathJax.typesetClear([eqn]);
      eqn.innerHTML = renderedEquation;
      eqn.dataset.renderedEquation = renderedEquation;
    }
  });

  MathJax.typesetPromise(eqnsToTypeset);
};

const sliders = document.querySelectorAll(".slider-container input");

sliders.forEach((slider) =>
  slider.addEventListener("input", () => update_centered_equations())
);

update_centered_equations();
