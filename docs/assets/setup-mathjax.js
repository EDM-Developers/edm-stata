window.MathJax = {
  loader: {load: ['[tex]/boldsymbol']},
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    packages: {'[+]': ['boldsymbol']}
  },
  startup: {
    typeset: true, // because we load MathJax asynchronously
  },
  options: {
    processHtmlClass: "arithmatex"
  },
  chtml: {
    mtextInheritFont: true,       // true to make mtext elements use surrounding font
  }
};
