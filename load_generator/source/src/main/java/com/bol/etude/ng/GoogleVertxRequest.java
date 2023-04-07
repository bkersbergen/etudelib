package com.bol.etude.ng;

import java.util.ArrayList;
import java.util.List;

public class GoogleVertxRequest {

    public List<Instance> instances = new ArrayList<>(1);

    public List<Parameter> parameters =  new ArrayList<>(1) {{
        add(new Parameter());
    }};

    GoogleVertxRequest(List<Long> values) {
        instances.add(new Instance(values));
    }

    static class Instance {
        final List<Long> context;

        Instance(List<Long> context) {
            this.context = context;
        }
    }

    static class Parameter {
        final String runtime  = "eager";
    }
}
