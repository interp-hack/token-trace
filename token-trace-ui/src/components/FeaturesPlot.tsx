import Plot from "react-plotly.js";

interface FeaturesPlotProps {
  vals: [number, number][];
  numFeatures: number;
  maxVal: number;
  minVal: number;
}

const FeaturesPlot = ({
  vals,
  numFeatures,
  maxVal,
  minVal,
}: FeaturesPlotProps) => {
  const xyVals: Map<number, number> = new Map();
  xyVals.set(numFeatures, 0);
  xyVals.set(0, 0);
  for (const [i, val] of vals) {
    if (i > 0 && !xyVals.has(i - 1)) {
      xyVals.set(i - 1, 0);
    }
    xyVals.set(i, val);
    xyVals.set(i + 1, 0);
  }
  const entries = Array.from(xyVals.entries()).sort((a, b) => a[0] - b[0]);

  const data = [
    {
      x: entries.map((xy) => xy[0]),
      y: entries.map((xy) => xy[1]),
      type: "line",
      mode: "lines",
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
    xaxis: {
      showgrid: false,
    },
    yaxis: {
      showgrid: false,
      range: [1.25 * minVal, 1.25 * maxVal],
    },
  };

  return (
    <Plot data={data} layout={layout} config={{ displayModeBar: false }} />
  );
};

export default FeaturesPlot;
