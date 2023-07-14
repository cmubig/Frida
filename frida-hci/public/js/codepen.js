function _defineProperty(obj, key, value) { key = _toPropertyKey(key); if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }
function _toPropertyKey(arg) { var key = _toPrimitive(arg, "string"); return typeof key === "symbol" ? key : String(key); }
function _toPrimitive(input, hint) { if (typeof input !== "object" || input === null) return input; var prim = input[Symbol.toPrimitive]; if (prim !== undefined) { var res = prim.call(input, hint || "default"); if (typeof res !== "object") return res; throw new TypeError("@@toPrimitive must return a primitive value."); } return (hint === "string" ? String : Number)(input); }
const cursor = document.getElementById("cursor");
const amount = 20;
const sineDots = Math.floor(amount * 0.3);
const width = 26;
const idleTimeout = 150;
let lastFrame = 0;
let mousePosition = {
  x: 0,
  y: 0
};
let dots = [];
let timeoutID;
let idle = false;
class Dot {
  constructor(index = 0) {
    this.index = index;
    this.anglespeed = 0.05;
    this.x = 0;
    this.y = 0;
    this.scale = 1 - 0.05 * index;
    this.range = width / 2 - width / 2 * this.scale + 2;
    this.limit = width * 0.75 * this.scale;
    this.element = document.createElement("span");
    TweenMax.set(this.element, {
      scale: this.scale
    });
    cursor.appendChild(this.element);
  }
  lock() {
    this.lockX = this.x;
    this.lockY = this.y;
    this.angleX = Math.PI * 2 * Math.random();
    this.angleY = Math.PI * 2 * Math.random();
  }
  draw(delta) {
    if (!idle || this.index <= sineDots) {
      TweenMax.set(this.element, {
        x: this.x,
        y: this.y
      });
    } else {
      this.angleX += this.anglespeed;
      this.angleY += this.anglespeed;
      this.y = this.lockY + Math.sin(this.angleY) * this.range;
      this.x = this.lockX + Math.sin(this.angleX) * this.range;
      TweenMax.set(this.element, {
        x: this.x,
        y: this.y
      });
    }
  }
}
function init() {
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("touchmove", onTouchMove);
  // eslint-disable-next-line no-new
  lastFrame += new Date();
  buildDots();
  render();
}

/*function limit(value, min, max) {
    return Math.min(Math.max(min, value), max);
}*/
function buildDots() {
  for (let i = 0; i < amount; i++) {
    let dot = new Dot(i);
    dots.push(dot);
  }
}
const onMouseMove = event => {
  mousePosition.x = event.clientX - width / 2;
  mousePosition.y = event.clientY - width / 2;
};
const onTouchMove = () => {
  mousePosition.x = event.touches[0].clientX - width / 2;
  mousePosition.y = event.touches[0].clientY - width / 2;
};
const render = timestamp => {
  const delta = timestamp - lastFrame;
  positionCursor(delta);
  lastFrame = timestamp;
  requestAnimationFrame(render);
};
const positionCursor = delta => {
  let x = mousePosition.x;
  let y = mousePosition.y;
  dots.forEach((dot, index, dots) => {
    let nextDot = dots[index + 1] || dots[0];
    dot.x = x;
    dot.y = y;
    dot.draw(delta);
    if (!idle || index <= sineDots) {
      const dx = (nextDot.x - dot.x) * 0.35;
      const dy = (nextDot.y - dot.y) * 0.35;
      x += dx;
      y += dy;
    }
  });
};
init();