import React, { useState } from "react";
import "./TokenTrace.css";
import Feature from "./Feature";

export interface FeatureVal {
  index: number;
  value: number;
}

export interface TokenTraceProps {
  tokens: string[];
  layerVals: FeatureVal[][][];
}

const TokenTrace = ({ tokens, layerVals }: TokenTraceProps) => {
  const [hlFeature, setHlFeature] = useState<[number, number] | null>(null);

  const numLayers = layerVals[0].length;
  return (
    <table className="TokenTrace">
      <tr className="TokenTrace-header">
        <th></th>
        {Array.from({ length: numLayers }, (_, i) => (
          <th key={i}>{i}</th>
        ))}
      </tr>
      {tokens.map((token, i) => (
        <tr key={i}>
          <td className="TokenTrace-token">{token}</td>
          {layerVals[i].map((featureVals, j) => (
            <td key={j} className="TokenTrace-td">
              <div className="TokenTrace-box">
                {featureVals.map((featureVal, k) => (
                  <Feature
                    key={k}
                    {...featureVal}
                    onMouseEnter={() => setHlFeature([j, featureVal.index])}
                    onMouseLeave={() => setHlFeature(null)}
                    highlight={
                      hlFeature !== null &&
                      hlFeature[0] === j &&
                      hlFeature[1] === featureVal.index
                    }
                  />
                ))}
              </div>
            </td>
          ))}
        </tr>
      ))}
    </table>
  );
};

export default TokenTrace;
