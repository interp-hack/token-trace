interface GradientRangeProps {
  maxValue: number;
}

const GradientRange = ({ maxValue }: GradientRangeProps) => {
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
      background: `linear-gradient(to right, rgba(123, 94, 17, 0) 0%, rgba(123, 94, 17, 1) 100%)`,
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
      <div style={styles.gradientRangeLabel}>0</div>
      <div style={styles.gradientRangeColor} />
      <div style={styles.gradientRangeLabel}>{maxValue.toFixed(0)}</div>
    </div>
  );
};

export default GradientRange;
