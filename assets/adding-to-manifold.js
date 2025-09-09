const update_centered_equations = function () {
  // Read in the manifold specifications from the sliders
  const numObs = parseInt(document.getElementById("numObs").value);
  const E = parseInt(document.getElementById("E").value);
  const tau = parseInt(document.getElementById("tau").value);
  const p = 1;
  const allowMissing = false;

  // Construct the manifold and targets
  const M_a = manifold("a", numObs, E, tau, allowMissing, p).manifold;
  const M_b = manifold("b", numObs, E, tau, allowMissing, p).manifold;

  let M_a_b = M_a.map((point) => point.slice());
  let M_a_b_varying = M_a.map((point) => point.slice());
  for (let i = 0; i < M_a.length; i++) {
    M_a_b[i].push(M_b[i][0]);
    for (let j = 0; j < M_b[i].length; j++) {
      M_a_b_varying[i].push(M_b[i][j]);
    }
  }

  // Turn these into latex arrays
  const M_a_tex = latexify(M_a);
  const M_a_b_tex = latexify(M_a_b);
  const M_a_b_varying_tex = latexify(M_a_b_varying);

  // Save the result to the page
  let eqnsToTypeset = [];

  const equations = document.querySelectorAll(".dynamic-equation");
  equations.forEach((eqn) => {
    const prevRenderedEquation = eqn.dataset.renderedEquation;
    const renderedEquation = eqn.dataset.equation
      .replace(/\${M_a}/, M_a_tex)
      .replace(/\${M_a_b}/, M_a_b_tex)
      .replace(/\${M_a_b_varying}/, M_a_b_varying_tex);

    if (prevRenderedEquation != renderedEquation) {
      eqnsToTypeset.push(eqn);
      if (MathJax.typesetClear) {
        MathJax.typesetClear([eqn]);
      }
      eqn.innerHTML = renderedEquation;
      eqn.dataset.renderedEquation = renderedEquation;
    }
  });

  if (MathJax.typesetPromise) {
    MathJax.typesetPromise(eqnsToTypeset);
  }
};

const sliders = document.querySelectorAll(".slider-container input");

sliders.forEach((slider) =>
  slider.addEventListener("input", () => update_centered_equations())
);

update_centered_equations();
