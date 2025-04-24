import React from "react";

export const PlusFilledIcon: React.FC<{
  color?: string;
  size?: number;
}> = ({ size = 20, color = "#9CA3AF" }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 20 20"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <g id="Plus">
        <path
          id="Shape"
          fillRule="evenodd"
          clipRule="evenodd"
          d="M10.5003 3.6665C10.5003 3.39036 10.2765 3.1665 10.0003 3.1665C9.72418 3.1665 9.50033 3.39036 9.50033 3.6665V9.49984H3.66699C3.39085 9.49984 3.16699 9.72369 3.16699 9.99984C3.16699 10.276 3.39085 10.4998 3.66699 10.4998H9.50033V16.3332C9.50033 16.6093 9.72418 16.8332 10.0003 16.8332C10.2765 16.8332 10.5003 16.6093 10.5003 16.3332V10.4998H16.3337C16.6098 10.4998 16.8337 10.276 16.8337 9.99984C16.8337 9.72369 16.6098 9.49984 16.3337 9.49984H10.5003V3.6665Z"
          fill={color}
        />
      </g>
    </svg>
  );
};
