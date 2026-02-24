import type { Site, Metadata, Socials } from "@types";

export const SITE: Site = {
  NAME: "Etornam",
  EMAIL: "navidben45@gmail.com",
  NUM_POSTS_ON_HOMEPAGE: 3,
  NUM_PAPERS_ON_HOMEPAGE: 2,
  NUM_PROJECTS_ON_HOMEPAGE: 3,
};

export const HOME: Metadata = {
  TITLE: "Home",
  DESCRIPTION: "My minimal blog and portfolio.",
};

export const BLOG: Metadata = {
  TITLE: "Blog",
  DESCRIPTION: "A collection of articles on topics I am passionate about.",
};

export const PAPERS: Metadata = {
  TITLE: "Papers",
  DESCRIPTION: "Papers I have read and my thoughts on them.",
};

export const PROJECTS: Metadata = {
  TITLE: "Projects",
  DESCRIPTION: "A collection of my projects, with links to repositories and demos.",
};

export const SOCIALS: Socials = [
  { 
    NAME: "twitter-x",
    HREF: "https://twitter.com/etornam45",
  },
  { 
    NAME: "github",
    HREF: "https://github.com/etornam45"
  },
  { 
    NAME: "linkedin",
    HREF: "https://www.linkedin.com/in/etornam45",
  }
];
