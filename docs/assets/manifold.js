function latexify(M) {
  if (M.length == 0) {
    return "\\emptyset";
  }

  const ncols = M[0].length;
  let tex = `\\left( \\begin{array}{${"c".repeat(ncols)}}\n`;

  for (let i = 0; i < M.length; i++) {
    tex += "\t";
    for (let j = 0; j < ncols; j++) {
      tex += `${M[i][j]}`;
      if (j < ncols - 1) {
        tex += " & ";
      }
    }
    tex += " \\\\ \n";
  }
  tex += "\\end{array} \\right)";
  return tex;
}

function manifold(x, numObs, E, tau, allowMissing = false, p = 1) {
  let M = [];
  if (E == 0) {
    return M;
  }

  let targets = [];

  for (let i = 1; i <= numObs; i++) {
    const targetInd = i + p;
    const targetMissing = targetInd < 1 || targetInd > numObs;
    const target = targetMissing ? "\\text{NA}" : `${x}_{${targetInd}}`;

    let M_i = [];
    let hasMissing = false;
    for (let j = 0; j < E; j++) {
      ind = i - j * tau;
      indMissing = ind < 1 || ind > numObs;
      if (indMissing) {
        M_i.push("\\text{NA}");
      } else {
        M_i.push(`${x}_{${ind}}`);
      }
      hasMissing |= indMissing;
    }

    if (hasMissing && !allowMissing) {
      continue;
    }
    M.push(M_i);
    targets.push([target]);
  }
  return { manifold: M, targets: targets };
}
