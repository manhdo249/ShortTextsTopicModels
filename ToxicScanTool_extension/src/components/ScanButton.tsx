import React from "react";
import { Button } from "@nextui-org/react";
import { twMerge } from "tailwind-merge";

interface ScanButtonProps {
  onClick: () => void;
  isLoading?: boolean;
  isDisabled?: boolean;
  scanBtnClassName?: string;
}

const ScanButton: React.FC<ScanButtonProps> = ({
  onClick,
  isLoading = false,
  isDisabled,
  scanBtnClassName,
}) => {
  return (
    <Button
      className={twMerge(
        "shadow-in h-10 w-full overflow-hidden rounded-full bg-aqua px-7 text-16 font-medium text-gray-950 shadow-4 transition-all duration-500 ease-linear",
        scanBtnClassName,
      )}
      onClick={onClick}
      isLoading={isLoading}
      isDisabled={isDisabled}
    >
      Scan Page
    </Button>
  );
};

export default ScanButton;
