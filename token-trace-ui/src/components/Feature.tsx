import classNames from "classnames";

export interface FeatureProps {
  index: number;
  value: number;
  highlight: boolean;
  maxValue: number;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

const Feature = ({
  index,
  value,
  highlight,
  maxValue,
  onMouseEnter,
  onMouseLeave,
}: FeatureProps) => {
  const borderColor = highlight ? "rgba(0, 0, 0, 0.5)" : "rgba(0, 0, 0, 0.0)";
  const portion = value / maxValue;
  const styles: any = {
    feature: {
      display: "flex",
      flexDirection: "column",
      fontSize: "7px",
      color: "#555",
      margin: "2px 4px",
      textAlign: "center",
      border: `1px solid ${borderColor}`,
    },
    featureValue: {
      width: "25px",
      height: "25px",
      margin: "2px",
      backgroundColor: "#EEE",
    },
    featureValueInner: {
      width: "25px",
      height: "25px",
      backgroundColor: `rgba(123, 94, 17, ${portion})`,
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      color: portion > 0.5 ? "#FFF" : "#000",
    },
  };

  return (
    <div
      style={styles.feature}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="Feature-value" style={styles.featureValue}>
        <div style={styles.featureValueInner}>
          {parseFloat(`${value}`).toFixed(2)}
        </div>
      </div>
      {index}
    </div>
  );
};

export default Feature;
