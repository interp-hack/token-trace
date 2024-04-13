/* eslint-disable import/no-extraneous-dependencies */
import React from "react";
import ReactDOM from "react-dom/client";

export function render(
  divID: string,
  circuitsVisElement: any,
  props: { [key: string]: any } = {}
) {
  const div = document.querySelector(`#${divID}`) as HTMLDivElement;
  const root = ReactDOM.createRoot(div);
  const element = React.createElement(circuitsVisElement, props);
  root.render(element);
}
