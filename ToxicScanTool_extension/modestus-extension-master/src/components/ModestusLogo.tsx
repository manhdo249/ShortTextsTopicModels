import { Image, Link } from "@nextui-org/react";
import { modestusLogo } from "../assets/icons";

const ModestusLogo = () => {
  return (
    <Link
      className="flex cursor-pointer items-center gap-[10px]"
      href="https://modestus.ai/"
      target="_blank"
    >
      <Image
        src={modestusLogo}
        alt="modestus logo"
        className="max-w-none rounded-none"
      />
      <span className="text-20 font-medium text-[#E9FBFB]">Modestus</span>
    </Link>
  );
};

export default ModestusLogo;
