import classNames from "classnames";

export interface FeatureProps {
  index: number;
  value: number;
  layer: number;
  highlight: boolean;
  maxValue: number;
  minValue: number;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

const Feature = ({
  index,
  value,
  layer,
  highlight,
  maxValue,
  minValue,
  onMouseEnter,
  onMouseLeave,
}: FeatureProps) => {
  const borderColor = highlight ? "rgba(0, 0, 0, 0.5)" : "rgba(0, 0, 0, 0.0)";
  const maxAbsVal = Math.max(Math.abs(maxValue), Math.abs(minValue));
  const portion = Math.abs(value / maxAbsVal);
  const styles: any = {
    feature: {
      display: "flex",
      flexDirection: "column",
      fontSize: "6px",
      color: "#555",
      margin: "1px 1px",
      textAlign: "center",
      border: `1px solid ${borderColor}`,
      textDecoration: "none",
    },
    featureValue: {
      width: "15px",
      height: "15px",
      margin: "2px",
      backgroundColor: "#EEE",
    },
    featureValueInner: {
      width: "15px",
      height: "15px",
      backgroundColor:
        value > 0
          ? `rgba(0, 0, 255, ${portion})`
          : `rgba(255, 0, 0, ${portion})`,
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      color: portion > 0.5 ? "#FFF" : "#000",
    },
  };

  const expVal = value.toExponential(0);
  const precVal = value.toPrecision(3);
  const valDisp = expVal.length < precVal.length ? expVal : precVal;

  return (
    <>
      <a
        style={styles.feature}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
        href={`https://www.neuronpedia.org/gpt2-small/${layer}-res-jb/${index}`}
      >
        <div className="Feature-value" style={styles.featureValue}>
          <div style={styles.featureValueInner}>{valDisp}</div>
        </div>
        {index}
      </a>
    </>
  );
};

export default Feature;
