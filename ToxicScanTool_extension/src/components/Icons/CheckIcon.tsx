import React from "react";

export const CheckFilledIcon: React.FC<{
  color?: string;
  size?: number;
}> = ({ size = 12, color = "#F3F4F6" }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 12 12"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <g id="Check">
        <path
          id="Shape"
          fillRule="evenodd"
          clipRule="evenodd"
          d="M9.17343 2.98147C9.40455 3.13259 9.46941 3.44246 9.31829 3.67358L5.91829 8.87358C5.83817 8.99613 5.70835 9.0774 5.56311 9.09594C5.41788 9.11447 5.27181 9.06842 5.16347 8.96993L2.96347 6.96993C2.75914 6.78418 2.74408 6.46795 2.92984 6.26362C3.11559 6.05929 3.43182 6.04424 3.63614 6.22999L5.40224 7.83553L8.48132 3.12633C8.63244 2.89521 8.94231 2.83036 9.17343 2.98147Z"
          fill={color}
        />
      </g>
    </svg>
  );
};
