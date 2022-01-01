update_manifold_specs = function () {
  document.querySelectorAll(".dynamic-inline").forEach((eqn) => {
    const E = parseInt(document.getElementById("E").value);
    const tau = parseInt(document.getElementById("tau").value);

    eqn.innerHTML =
      "\\(" +
      eqn.dataset.equation.replace(/\${E}/, E).replace(/\${tau}/, tau) +
      "\\)";
  });
};

update_manifold = function () {
  // Read in the manifold specifications from the sliders
  const numObs = parseInt(document.getElementById("numObs").value);
  const E = parseInt(document.getElementById("E").value);
  const tau = parseInt(document.getElementById("tau").value);
  // let allowMissing = document.getElementById("allowMissing").checked
  const allowMissing = false;
  const p = 1;

  // Construct the manifold and targets
  const M = manifold("x", numObs, E, tau, allowMissing, p);

  // Turn these into latex arrays
  const maniSetFormTex = latexify_set_of_sets(M.manifold);
  const maniTex = latexify(M.manifold);
  // const targetsTex = latexify(M.targets);

  // Save the result to the page
  document.querySelectorAll(".dynamic-equation").forEach((eqn) => {
    eqn.innerHTML = eqn.dataset.equation
      .replace(/\${M_x_sets}/, maniSetFormTex)
      .replace(/\${M_x}/, maniTex);
  });

  MathJax.typeset();
};

const sliderIDs = ["numObs", "E", "tau"];
for (let sliderID of sliderIDs) {
  let slider = document.getElementById(sliderID);

  // Display the default slider value
  document
    .querySelectorAll(`.${sliderID}_choice`)
    .forEach((elem) => (elem.innerHTML = `${slider.value}`));

  // Update the current slider value (each time you drag the slider handle)
  slider.oninput = function () {
    document
      .querySelectorAll(`.${this.id}_choice`)
      .forEach((elem) => (elem.innerHTML = `${this.value}`));

    update_manifold_specs();
    update_manifold();
  };
}

// document.getElementById("allowMissing").oninput = update_manifold
update_manifold_specs();
update_manifold();
