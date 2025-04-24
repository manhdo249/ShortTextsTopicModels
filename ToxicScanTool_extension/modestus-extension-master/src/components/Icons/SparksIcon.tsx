import React from "react";

export const SparksFilledIcon: React.FC<{
  color?: string;
  size?: number;
}> = ({ size = 20, color = "white" }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <g id="Sparks">
        <path
          id="Vector"
          d="M7.06689 12.1997C10.6417 12.1997 12.2002 10.6957 12.2002 7.06641C12.2002 10.6957 13.7479 12.1997 17.3336 12.1997C13.7479 12.1997 12.2002 13.7474 12.2002 17.3331C12.2002 13.7474 10.6417 12.1997 7.06689 12.1997Z"
          fill={color}
          stroke={color}
          strokeWidth="1.1"
          strokeLinejoin="round"
        />
        <path
          id="Vector_2"
          d="M2.6665 5.9665C4.96459 5.9665 5.9665 4.99963 5.9665 2.6665C5.9665 4.99963 6.96141 5.9665 9.2665 5.9665C6.96141 5.9665 5.9665 6.96141 5.9665 9.2665C5.9665 6.96141 4.96459 5.9665 2.6665 5.9665Z"
          fill={color}
          stroke={color}
          strokeWidth="1.1"
          strokeLinejoin="round"
        />
      </g>
    </svg>
  );
};
