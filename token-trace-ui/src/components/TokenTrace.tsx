import React, { useState } from "react";
import Feature from "./Feature";
import FeaturesPlot from "./FeaturesPlot";
import GradientRange from "./GradientRange";
import { unionLayerVals } from "../unionLayerVals";
import { filterLayerVals } from "../filterLayerVals";

export interface TokenTraceProps {
  tokens: string[];
  layerVals: [number, number][][][];
  numFeatures?: number;
  hideBoxes?: boolean;
  hideBars?: boolean;
  unionColToks?: boolean;
  filterBottomPortion?: number;
}

const TokenTrace = ({
  tokens,
  layerVals,
  numFeatures = 24576,
  hideBoxes = false,
  hideBars = false,
  unionColToks = false,
  filterBottomPortion = 0,
}: TokenTraceProps) => {
  const [hlFeature, setHlFeature] = useState<[number, number] | null>(null);
  const valsPerCol: { [key: number]: number[] } = {};
  for (const tokenVals of layerVals) {
    tokenVals.map((colVals, i) => {
      if (!valsPerCol[i]) {
        valsPerCol[i] = [];
      }
      for (const featureVal of colVals) {
        valsPerCol[i].push(featureVal[1]);
      }
    });
  }
  const maxValsPerCol = Object.values(valsPerCol).map((vals) =>
    Math.max(Math.max(...vals), 0)
  );
  const minValsPerCol = Object.values(valsPerCol).map((vals) =>
    Math.min(Math.min(...vals), 0)
  );
  const maxAbsValsPerCol = maxValsPerCol.map((maxVal, i) => {
    const minVal = minValsPerCol[i];
    return Math.max(Math.abs(maxVal), Math.abs(minVal));
  });

  const styles: any = {
    tokenTrace: {
      fontSize: "14px",
      fontFamily: "Arial, Helvetica, sans-serif",
      backgroundColor: "#f9f9f9",
      padding: "10px",
      color: "#333",
      maxWidth: "100%",
      overflowX: "auto",
    },
    tokenTraceTable: {
      width: `${200 * (layerVals[0].length + 1)}px`,
    },
    tokenTraceHeader: {
      alignItems: "center",
      textAlign: "center",
      fontWeight: "light", // This might need to be a numeric value like 300 if "light" does not work
      fontSize: "12px",
    },
    tokenTraceToken: {
      textAlign: "right",
      padding: "5px",
    },
    tokenTraceTd: {
      width: "200px",
      border: "1px solid #e0e0e0",
      height: "inherit",
    },
    tokenTraceTr: {
      height: "1px", // hacky, from https://stackoverflow.com/questions/3215553/make-a-div-fill-an-entire-table-cell
    },
    tokenTraceBox: {
      display: "flex",
      flexWrap: "wrap",
      padding: "2px",
      margin: "2px",
    },
    tokenTraceBoxOuter: {
      display: "flex",
      flexDirection: "column",
      height: "100%",
      justifyContent: "space-between",
    },
  };

  const numLayers = layerVals[0].length;
  let processedLayers = layerVals;
  if (filterBottomPortion > 0) {
    const minAbsValPerCol = maxAbsValsPerCol.map(
      (maxAbsVal) => maxAbsVal * filterBottomPortion
    );
    processedLayers = filterLayerVals(layerVals, minAbsValPerCol);
  }
  if (unionColToks) {
    processedLayers = unionLayerVals(processedLayers);
  }

  return (
    <div style={styles.tokenTrace}>
      <table style={styles.tokenTraceTable}>
        <tr>
          <th style={styles.tokenTraceHeader}></th>
          {Array.from({ length: numLayers }, (_, i) => (
            <th key={i} style={styles.tokenTraceHeader}>
              {i}
            </th>
          ))}
        </tr>
        <tr>
          <td></td>
          {maxValsPerCol.map((maxVal, i) => (
            <td key={i}>
              <GradientRange maxValue={maxVal} minValue={minValsPerCol[i]} />
            </td>
          ))}
        </tr>
        {tokens.map((token, i) => (
          <tr key={i} style={styles.tokenTraceTr}>
            <td style={styles.tokenTraceToken}>{token}</td>
            {processedLayers[i].map((featureVals, j) => (
              <td key={j} style={styles.tokenTraceTd}>
                <div style={styles.tokenTraceBoxOuter}>
                  {!hideBoxes && (
                    <div style={styles.tokenTraceBox}>
                      {featureVals.map((featureVal, k) => (
                        <Feature
                          key={k}
                          index={featureVal[0]}
                          value={featureVal[1]}
                          layer={j}
                          maxValue={maxValsPerCol[j]}
                          minValue={minValsPerCol[j]}
                          onMouseEnter={() => setHlFeature([j, featureVal[0]])}
                          onMouseLeave={() => setHlFeature(null)}
                          highlight={
                            hlFeature !== null &&
                            hlFeature[0] === j &&
                            hlFeature[1] === featureVal[0]
                          }
                        />
                      ))}
                    </div>
                  )}
                  {!hideBars && (
                    <div>
                      <FeaturesPlot
                        vals={featureVals}
                        numFeatures={numFeatures}
                        maxVal={maxValsPerCol[j]}
                        minVal={minValsPerCol[j]}
                      />
                    </div>
                  )}
                </div>
              </td>
            ))}
          </tr>
        ))}
      </table>
    </div>
  );
};

export default TokenTrace;
