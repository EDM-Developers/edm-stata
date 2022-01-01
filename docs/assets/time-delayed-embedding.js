update_manifold = function (refresh = true) {
  // Read in the manifold specifications from the sliders
  let numObs = parseInt(document.getElementById("numObs").value);
  let E = parseInt(document.getElementById("E").value);
  let tau = parseInt(document.getElementById("tau").value);
  // let allowMissing = document.getElementById("allowMissing").checked
  // let p = parseInt(document.getElementById("p").value)
  let allowMissing = false;
  let p = 1;

  // Construct the manifold and targets
  let M = manifold("x", numObs, E, tau, allowMissing, p);

  // Turn these into latex arrays
  let maniTex = latexify(M.manifold);
  let targetsTex = latexify(M.targets);

  // Save the result to the page
  document.getElementById(
    "manifold"
  ).innerHTML = `\\[ M_x = ${maniTex}, \\quad y = ${targetsTex} \\]`;

  document.querySelectorAll(".dynamic-equation").forEach((eqn) => {
    const tex = eqn.dataset.equation;
    if (tex.contains("${M_x}") || tex.contains("${y}")) {
      eqn.innerHTML = tex.replace(/\${M_x}/, maniTex).replace(/\${y}/, y);
    }
  });

  if (refresh) {
    MathJax.typeset();
  }
};

const sliderIDs = ["numObs", "E", "tau"]; //, "p"]
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
update_manifold(refresh = false);