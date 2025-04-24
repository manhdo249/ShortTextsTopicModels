const path = require("path");
const HTMLPlugin = require("html-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  entry: {
    index: path.join(__dirname, "src/index.tsx"),
    contentScript: path.join(__dirname, "src/contentScript.ts"),
    background: path.join(__dirname, "src/background.ts"),
  },
  output: { path: path.join(__dirname, "dist"), filename: "[name].js" },
  mode: "production",
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: [
          {
            loader: "ts-loader",
            options: {
              compilerOptions: { noEmit: false },
            },
          },
        ],
        exclude: /node_modules/,
      },
      {
        test: /\.(gif|svg)$/i,
        use: [
          "@svgr/webpack",
          {
            loader: "url-loader",
            options: {
              limit: 8192,
            },
          },
        ],
      },
      {
        test: /\.(png|jpg)$/i,
        use: [
          {
            loader: "url-loader",
            options: {
              mimetype: "image/png",
            },
          },
        ],
      },
      {
        exclude: /node_modules/,
        test: /\.css$/i,
        use: ["style-loader", "css-loader", "postcss-loader"],
      },
    ],
  },
  plugins: [
    new CopyPlugin({
      patterns: [{ from: "public", to: "." }],
    }),
  ],
  devServer: {
    contentBase: "./dist",
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
  },
};

// function getHtmlPlugins(chunks) {
//   return chunks.map(
//     (chunk) =>
//       new HTMLPlugin({
//         title: "React extension",
//         filename: `${chunk}.html`,
//         chunks: [chunk],
//       }),
//   );
// }
