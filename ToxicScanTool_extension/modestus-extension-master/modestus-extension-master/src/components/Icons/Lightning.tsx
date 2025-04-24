import React from "react";

export const LightningBoltIcon: React.FC<{
  color?: string;
  size?: number;
}> = ({ size = 16, color = "#9CA3AF" }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M7.99984 6.9332L9.0665 0.533203L2.6665 9.06654H7.99984L6.93317 15.4665L13.3332 6.9332H7.99984Z"
        fill={color}
      />
    </svg>
  );
};
