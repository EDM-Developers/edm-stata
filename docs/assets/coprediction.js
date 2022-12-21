const update_centered_equations = function () {
  // Read in the manifold specifications from the sliders
  const numObs = parseInt(document.getElementById("numObs").value);
  const E = parseInt(document.getElementById("E").value);
  const tau = parseInt(document.getElementById("tau").value);
  const allowMissing = false;

  // Construct the manifold and targets
  const p = 1;
  const {manifold: M_a, targets: y_L_a} = manifold("a", numObs, E, tau, allowMissing, p);
  const {manifold: M_c, targets: y_P_c} = manifold("c", numObs, E, tau, allowMissing, p);

  const p_xmap = 0;
  const M_a_xmap = manifold("a", numObs, E, tau, allowMissing, p_xmap).manifold;
  const y_L_b_xmap = manifold("b", numObs, E, tau, allowMissing, p_xmap).targets;
  const {manifold: M_c_xmap, targets: y_P_c_xmap} = manifold("c", numObs, E, tau, allowMissing, p_xmap);

  // Save the result to the page
  let eqnsToTypeset = [];

  const equations = document.querySelectorAll(".dynamic-equation");
  equations.forEach((eqn) => {
    const prevRenderedEquation = eqn.dataset.renderedEquation;
    const renderedEquation = eqn.dataset.equation
      .replace(/\${M_a}/, latexify(M_a))
      .replace(/\${y_L_a}/, latexify(y_L_a))
      .replace(/\${M_c}/, latexify(M_c))
      .replace(/\${y_P_c}/, latexify(y_P_c))
      .replace(/\${M_a_xmap}/, latexify(M_a_xmap))
      .replace(/\${y_L_b_xmap}/, latexify(y_L_b_xmap))
      .replace(/\${M_c_xmap}/, latexify(M_c_xmap))
      .replace(/\${y_P_c_xmap}/, latexify(y_P_c_xmap));

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
