package nl.peterbloem.util;

import java.util.List;

/**
 * Represents a class which can generate random real valued multivariate points.
 * 
 * @author Peter
 */
public interface Generator<P>
{
	public P generate();	
	
	public List<P> generate(int n);
}
