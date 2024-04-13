import Plot from "react-plotly.js";

interface FeaturesPlotProps {
  vals: [number, number][];
  numFeatures: number;
}

const FeaturesPlot = ({ vals, numFeatures }: FeaturesPlotProps) => {
  const xVals = [...Array(numFeatures).keys()];
  const yVals = xVals.map(() => 0);
  for (const [i, val] of vals) {
    yVals[i] = val;
  }

  const data = [
    {
      x: xVals,
      y: yVals,
      type: "line",
    },
  ];

  const layout = {
    width: 180,
    height: 50,
    margin: {
      l: 0, // left margin
      r: 0, // right margin
      t: 0, // top margin
      b: 0, // bottom margin
      pad: 0, // padding between plot area and axis lines
    },
  };

  return (
    <Plot data={data} layout={layout} config={{ displayModeBar: false }} />
  );
};

export default FeaturesPlot;
