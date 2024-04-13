import React, { useState } from "react";
import Feature from "./Feature";

export interface TokenTraceProps {
  tokens: string[];
  layerVals: [number, number][][][];
}

const TokenTrace = ({ tokens, layerVals }: TokenTraceProps) => {
  const [hlFeature, setHlFeature] = useState<[number, number] | null>(null);

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
    },
    tokenTraceBox: {
      display: "flex",
      flexWrap: "wrap",
      padding: "2px",
      margin: "2px",
    },
  };

  const numLayers = layerVals[0].length;
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
        {tokens.map((token, i) => (
          <tr key={i}>
            <td style={styles.tokenTraceToken}>{token}</td>
            {layerVals[i].map((featureVals, j) => (
              <td key={j} style={styles.tokenTraceTd}>
                <div style={styles.tokenTraceBox}>
                  {featureVals.map((featureVal, k) => (
                    <Feature
                      key={k}
                      index={featureVal[0]}
                      value={featureVal[1]}
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
              </td>
            ))}
          </tr>
        ))}
      </table>
    </div>
  );
};

export default TokenTrace;
