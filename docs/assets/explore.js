update_manifold = function () {
  // Read in the manifold specifications from the sliders
  const numObs = parseInt(document.getElementById("numObs").value);
  const E = parseInt(document.getElementById("E").value);
  const tau = parseInt(document.getElementById("tau").value);
  // const allowMissing = document.getElementById("allowMissing").checked
  const p = parseInt(document.getElementById("p").value);
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

  const libFirstTex = latexify(libSet.slice(0, 1));
  const predFirstTex = latexify(predSet.slice(0, 1));
  const predFirstTargetTex = latexify(predTargets.slice(0, 1));

  const libTargetsTex = latexify(libTargets);
  const predTargetsTex = latexify(predTargets);
  const weightedSum =
    predSet.length > 0
      ? predSet[0].map((v, i) => `w_{1,${i}} \\times ${v}`).join(" + ")
      : "\\text{NA}";

  // Save the result to the page
  document.querySelectorAll(".dynamic-equation").forEach((eqn) => {
    eqn.innerHTML = eqn.dataset.equation
      .replace(/\${M_x}/, maniTex)
      .replace(/\${L}/, libSetTex)
      .replace(/\${P}/, predSetTex)
      .replace(/\${L_1}/, libFirstTex)
      .replace(/\${P_1}/, predFirstTex)
      .replace(/\${y_P_1}/, predFirstTargetTex)
      .replace(/\${y_L}/, libTargetsTex)
      .replace(/\${y_P}/, predTargetsTex)
      .replace(/\${yhat_P_1}/, weightedSum);
  });

  MathJax.typeset();
};

const sliderIDs = ["numObs", "E", "tau", "p"];
for (let sliderID of sliderIDs) {
  let slider = document.getElementById(sliderID);

  // Display the default slider value
  document
    .querySelectorAll(`.${sliderID}_choice`)
    .forEach((elem) => (elem.innerHTML = `${slider.value}`));

  // Update the current slider value (each time you drag the slider handle)
  slider.oninput = function (refresh = true) {
    document
      .querySelectorAll(`.${this.id}_choice`)
      .forEach((elem) => (elem.innerHTML = `${this.value}`));
    update_manifold();
  };
}

// document.getElementById("allowMissing").oninput = update_manifold
update_manifold();
