export const unionLayerVals = (
  layerVals: [number, number][][][]
): [number, number][][][] => {
  const indicesPerCol: { [key: number]: Set<number> } = {};
  for (const tokenVals of layerVals) {
    tokenVals.map((colVals, i) => {
      if (!indicesPerCol[i]) {
        indicesPerCol[i] = new Set();
      }
      for (const featureVal of colVals) {
        indicesPerCol[i].add(featureVal[0]);
      }
    });
  }
  // sort the indices
  const sortedIndicesPerCol: { [key: number]: number[] } = {};

  Object.entries(indicesPerCol).forEach(([col, indices]) => {
    sortedIndicesPerCol[parseInt(col)] = Array.from(indices).sort(
      (a, b) => a - b
    );
  });

  const newLayerVals: [number, number][][][] = [];
  for (const tokenVals of layerVals) {
    const newTokenVals: [number, number][][] = [];
    tokenVals.map((colVals, i) => {
      const newColVals: [number, number][] = [];
      sortedIndicesPerCol[i].map((index) => {
        newColVals.push(colVals.find((val) => val[0] === index) || [index, 0]);
      });
      newTokenVals.push(newColVals);
    });
    newLayerVals.push(newTokenVals);
  }

  return newLayerVals;
};
