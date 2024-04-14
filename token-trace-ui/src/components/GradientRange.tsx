interface GradientRangeProps {
  maxValue: number;
  minValue: number;
}

const GradientRange = ({ maxValue, minValue }: GradientRangeProps) => {
  const maxAbsVal = Math.max(Math.abs(maxValue), Math.abs(minValue));

  const styles: any = {
    gradientRange: {
      display: "flex",
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "center",
      backgroundColor: "#f9f9f9",
      padding: "5px 15px",
    },
    gradientRangeColor: {
      height: "5px",
      flexGrow: 1,
      // 0 should be full transparency and in the middle
      // negative is red, positive is blue
      background: `linear-gradient(to right, rgba(255, 0, 0, 1), rgba(255, 0, 0, 0), rgba(0, 0, 255, 1))`,
      border: "1px solid #e0e0e0",
    },
    gradientRangeLabel: {
      fontSize: "10px",
      color: "#333",
      margin: "0 5px",
    },
  };

  return (
    <div style={styles.gradientRange}>
      <div style={styles.gradientRangeLabel}>{(-1 * maxAbsVal).toFixed(2)}</div>
      <div style={styles.gradientRangeColor} />
      <div style={styles.gradientRangeLabel}>{maxAbsVal.toFixed(2)}</div>
    </div>
  );
};

export default GradientRange;
