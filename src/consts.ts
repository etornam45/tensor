import type { Site, Metadata, Socials } from "@types";

export const SITE: Site = {
  NAME: "Etornam",
  EMAIL: "navidben45@gmail.com",
  NUM_POSTS_ON_HOMEPAGE: 3,
  NUM_PAPERS_ON_HOMEPAGE: 2,
  NUM_PROJECTS_ON_HOMEPAGE: 3,
};

export const HOME: Metadata = {
  TITLE: "Minimalist Engineer & Problem Solver",
  DESCRIPTION: "Etornam is a minimalist engineer focused on solving real-world problems through robust system design and open-source contributions.",
};

export const BLOG: Metadata = {
  TITLE: "Blog | Insights on Engineering & Tech",
  DESCRIPTION: "A collection of articles and insights on topics I am passionate about, ranging from minimalist engineering to modern technology.",
};

export const PAPERS: Metadata = {
  TITLE: "Paper Reviews | Academic Insights",
  DESCRIPTION: "Detailed reviews and thoughts on academic papers that I have read and found interesting.",
};

export const PROJECTS: Metadata = {
  TITLE: "Projects | Built with Purpose",
  DESCRIPTION: "A curated collection of my projects, including repositories, demos, and the philosophy behind them.",
};

export const SOCIALS: Socials = [
  {
    NAME: "github",
    HREF: "https://github.com/etornam45"
  },
  {
    NAME: "linkedin",
    HREF: "https://www.linkedin.com/in/etornam45",
  },
  {
    NAME: "hugging-face",
    HREF: "https://www.huggingface.co/etornam",
  },
  {
    NAME: "twitter-x",
    HREF: "https://twitter.com/etornam45",
  },
];
