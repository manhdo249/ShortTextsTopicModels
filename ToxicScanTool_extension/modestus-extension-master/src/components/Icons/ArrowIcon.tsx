import React from "react";

export const ArrowRightFilledIcon: React.FC<{
  color?: string;
  size?: number;
}> = ({ size = 24, color = "#4B5563" }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <g id="Arrow Right">
        <path
          id="Shape"
          fillRule="evenodd"
          clipRule="evenodd"
          d="M13.0345 5.03461C13.3469 4.72219 13.8535 4.72219 14.1659 5.03461L20.5659 11.4346C20.8783 11.747 20.8783 12.2536 20.5659 12.566L14.1659 18.966C13.8535 19.2784 13.3469 19.2784 13.0345 18.966C12.7221 18.6536 12.7221 18.147 13.0345 17.8346L18.0688 12.8003H4.0002C3.55837 12.8003 3.2002 12.4421 3.2002 12.0003C3.2002 11.5585 3.55837 11.2003 4.0002 11.2003H18.0688L13.0345 6.16598C12.7221 5.85356 12.7221 5.34703 13.0345 5.03461Z"
          fill={color}
        />
      </g>
    </svg>
  );
};
