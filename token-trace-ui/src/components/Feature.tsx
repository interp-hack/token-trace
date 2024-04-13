import "./Feature.css";

import classNames from "classnames";

export interface FeatureProps {
  index: number;
  value: number;
  highlight: boolean;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

const Feature = ({
  index,
  value,
  highlight,
  onMouseEnter,
  onMouseLeave,
}: FeatureProps) => {
  return (
    <div
      className={classNames("Feature", { "Feature-highlight": highlight })}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div
        className="Feature-value"
        style={{
          backgroundColor: `rgba(123, 94, 17, ${value})`,
        }}
      />
      {index}
    </div>
  );
};

export default Feature;
