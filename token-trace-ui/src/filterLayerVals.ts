export const filterLayerVals = (
  layerVals: [number, number][][][],
  minAbsValPerCol: number[]
): [number, number][][][] => {
  const newLayerVals: [number, number][][][] = [];
  for (const tokenVals of layerVals) {
    const newTokenVals: [number, number][][] = [];
    tokenVals.map((colVals, i) => {
      const newColVals: [number, number][] = [];
      colVals.map((val) => {
        if (Math.abs(val[1]) >= minAbsValPerCol[i]) {
          newColVals.push(val);
        }
      });
      newTokenVals.push(newColVals);
    });
    newLayerVals.push(newTokenVals);
  }

  return newLayerVals;
};
